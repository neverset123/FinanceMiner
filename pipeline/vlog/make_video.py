"""
Core library for the vlog video generation pipeline.

Covers three responsibilities formerly split across multiple modules:

Image fetching (Wikimedia Commons)
    fetch_images() searches Commons by keyword and caches downloaded images
    under assets/images/. Up to 6 images per scene are fetched and composed
    into a responsive grid layout. Lines in the script may specify an explicit
    search query using the @@ separator; otherwise the text itself is used.

HTML scene composition
    _scene_html() builds a self-contained HTML page per scene: a CSS grid of
    background images (1 → full-cover, 2 → side-by-side, 3 → top+two, 4 → 2×2)
    with a Ken Burns pan/zoom animation and a radial vignette overlay.
    No text is rendered on-screen; narration is audio-only and captions go in
    the companion .srt file.

Video encoding
    generate_video() orchestrates the full render:
      1. Synthesise per-scene voiceover via edge-tts (online), pyttsx3
         (offline fallback), or a silent dummy WAV for testing.
      2. Fetch background images from Wikimedia Commons.
      3. Render each HTML scene to a 1280×720 PNG via headless Chromium
         (Playwright).
      4. Encode each PNG for its audio duration, concatenate all segments,
         mux with audio → MP4, and write an .srt subtitle file.

Entry point: run_pipeline.py (parses the script, calls extract_keywords for
LLM-based image queries, then calls generate_video here).

Requirements: requests, playwright, edge-tts, ffmpeg on PATH.
First run:    python -m playwright install chromium
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path

import requests

try:
    import edge_tts
except Exception:
    edge_tts = None

# ---------------------------------------------------------------------------
# Image fetching (Wikimedia Commons)
# ---------------------------------------------------------------------------

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
_HEADERS = {"User-Agent": "context-video-generator/1.0 (educational demo)"}

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None
        if proxy:
            s.proxies = {"http": proxy, "https": proxy}
        s.verify = os.getenv("NODE_TLS_REJECT_UNAUTHORIZED", "1") != "0"
        s.headers.update(_HEADERS)
        _session = s
    return _session


CACHE = Path("assets/images")
CACHE.mkdir(parents=True, exist_ok=True)

DEFAULT_QUERY = "global finance economy"


def query_for(text: str, explicit: str | None) -> str:
    """Determine search query: explicit @@ override, cleaned text, or default."""
    if explicit:
        return explicit
    clean = re.sub(r'[^\w\s]', ' ', text).strip()
    if clean and len(clean) > 2:
        clean = re.sub(r'^(第[一二三四五六七八九十]+|哈佬|大家好|歡迎|本期)', '', clean).strip()
        if clean:
            return clean[:50]
    return DEFAULT_QUERY


def _commons_search(query: str, limit: int = 8) -> list[str]:
    params = {
        "action": "query", "format": "json", "list": "search",
        "srsearch": f"{query} filetype:bitmap",
        "srnamespace": "6", "srlimit": str(limit),
    }
    r = _get_session().get(COMMONS_API, params=params, timeout=30)
    if r.status_code != 200:
        print(f"    [!] Commons search returned {r.status_code} for '{query}'")
        return []
    return [item["title"] for item in r.json().get("query", {}).get("search", [])]


def _thumb_url(title: str, width: int = 1280) -> str | None:
    params = {
        "action": "query", "format": "json", "prop": "imageinfo",
        "iiprop": "url|mime", "iiurlwidth": str(width), "titles": title,
    }
    r = _get_session().get(COMMONS_API, params=params, timeout=30)
    if r.status_code != 200:
        return None
    for p in r.json().get("query", {}).get("pages", {}).values():
        info = p.get("imageinfo", [{}])[0]
        mime = info.get("mime", "")
        if mime.startswith("image/") and mime != "image/svg+xml":
            return info.get("thumburl") or info.get("url")
    return None


def fetch_images(text: str, explicit: str | None = None, width: int = 1280,
                 count: int = 4) -> tuple[list[str], str]:
    """Return (list_of_local_paths, query_used). Fetches up to *count* images."""
    query = query_for(text, explicit)
    base_key = hashlib.md5(query.encode("utf-8")).hexdigest()[:12]

    # Return from cache if complete
    cached: list[str] = []
    for idx in range(count):
        suffix = "" if idx == 0 else f"_{idx}"
        for ext in (".jpg", ".jpeg", ".png"):
            p = CACHE / f"{base_key}{suffix}{ext}"
            if p.exists():
                cached.append(str(p))
                break
        else:
            break
    if len(cached) >= count:
        return cached, query

    paths: list[str] = []
    try:
        for title in _commons_search(query, limit=count + 4):
            if len(paths) >= count:
                break
            url = _thumb_url(title, width)
            if not url:
                continue
            resp = _get_session().get(url, timeout=40)
            if resp.status_code != 200 or not resp.content:
                continue
            ext = ".png" if url.lower().split("?")[0].endswith(".png") else ".jpg"
            suffix = "" if len(paths) == 0 else f"_{len(paths)}"
            out = CACHE / f"{base_key}{suffix}{ext}"
            out.write_bytes(resp.content)
            paths.append(str(out))
    except Exception as e:
        print(f"    [!] image fetch failed for '{query}': {type(e).__name__}")

    return paths, query

W, H = 1280, 720
FPS = 30

# Design tokens
BG_COLOR = "#111827"


def read_script(path: str) -> list[tuple[str, str | None]]:
    """Return list of (spoken_text, explicit_image_query_or_None).

    A line may specify an image search query after `@@`, e.g.
        text  @@  tokyo stock exchange, yen banknote
    """
    lines: list[tuple[str, str | None]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "@@" in s:
            text, q = s.split("@@", 1)
            lines.append((text.strip(), q.strip() or None))
        else:
            lines.append((s, None))
    if not lines:
        raise SystemExit(f"No usable lines found in {path}")
    return lines


# ---------------------------------------------------------------------------
# HTML Composition (similar to HyperFrames HTML-based scene rendering)
# ---------------------------------------------------------------------------

def _scene_html(text: str, index: int, total: int, bg_image_paths: list[str] | str | None = None) -> str:
    """Generate an HTML composition for one scene.

    The HTML includes:
    - A collage grid of background images with Ken Burns animation
      (4 → 2x2, 3 → 1 top + 2 bottom, 2 → side-by-side, 1 → full cover)
    - Subtle gradient vignette for cinematic feel
    - No on-screen text (narration is audio-only; subtitles go in .srt)
    """
    # Normalize input to a list of existing paths
    if isinstance(bg_image_paths, str):
        bg_image_paths = [bg_image_paths]
    paths: list[str] = []
    if bg_image_paths:
        for p in bg_image_paths:
            if p and os.path.exists(p):
                paths.append(os.path.abspath(p).replace("\\", "/"))

    n = len(paths)

    # Build grid cells HTML and CSS
    if n == 0:
        # No images — solid dark background
        grid_css = f"""
.grid {{
    position: absolute; inset: 0;
    background: {BG_COLOR};
}}"""
        grid_html = '<div class="grid"></div>'
    elif n == 1:
        # Single image — full cover (original behavior)
        grid_css = f"""
.grid {{
    position: absolute; inset: 0;
    display: grid;
    grid-template: 1fr / 1fr;
    gap: 0;
}}
.cell {{
    overflow: hidden;
}}
.cell img {{
    width: 100%; height: 100%;
    object-fit: cover;
    animation: kenburns 8s ease-in-out infinite alternate;
}}"""
        grid_html = f'''<div class="grid">
    <div class="cell"><img src="file:///{paths[0]}"></div>
</div>'''
    elif n == 2:
        # Side-by-side
        grid_css = f"""
.grid {{
    position: absolute; inset: 0;
    display: grid;
    grid-template: 1fr / 1fr 1fr;
    gap: 5px;
    background: #0a0a0a;
}}
.cell {{
    overflow: hidden;
}}
.cell img {{
    width: 100%; height: 100%;
    object-fit: cover;
    animation: kenburns 8s ease-in-out infinite alternate;
}}
.cell:nth-child(2) img {{
    animation-delay: 0.5s;
}}"""
        grid_html = f'''<div class="grid">
    <div class="cell"><img src="file:///{paths[0]}"></div>
    <div class="cell"><img src="file:///{paths[1]}"></div>
</div>'''
    elif n == 3:
        # 1 large top + 2 small bottom
        grid_css = f"""
.grid {{
    position: absolute; inset: 0;
    display: grid;
    grid-template-rows: 1fr 1fr;
    grid-template-columns: 1fr 1fr;
    gap: 5px;
    background: #0a0a0a;
}}
.cell {{
    overflow: hidden;
}}
.cell:first-child {{
    grid-column: 1 / -1;
}}
.cell img {{
    width: 100%; height: 100%;
    object-fit: cover;
    animation: kenburns 8s ease-in-out infinite alternate;
}}
.cell:nth-child(2) img {{
    animation-delay: 0.4s;
}}
.cell:nth-child(3) img {{
    animation-delay: 0.8s;
}}"""
        grid_html = f'''<div class="grid">
    <div class="cell"><img src="file:///{paths[0]}"></div>
    <div class="cell"><img src="file:///{paths[1]}"></div>
    <div class="cell"><img src="file:///{paths[2]}"></div>
</div>'''
    else:
        # 4+ images → 2x2 grid (use first 4)
        use = paths[:4]
        grid_css = f"""
.grid {{
    position: absolute; inset: 0;
    display: grid;
    grid-template: 1fr 1fr / 1fr 1fr;
    gap: 5px;
    background: #0a0a0a;
}}
.cell {{
    overflow: hidden;
}}
.cell img {{
    width: 100%; height: 100%;
    object-fit: cover;
    animation: kenburns 8s ease-in-out infinite alternate;
}}
.cell:nth-child(2) img {{ animation-delay: 0.3s; }}
.cell:nth-child(3) img {{ animation-delay: 0.6s; }}
.cell:nth-child(4) img {{ animation-delay: 0.9s; }}"""
        cells = "\n    ".join(
            f'<div class="cell"><img src="file:///{p}"></div>' for p in use
        )
        grid_html = f'<div class="grid">\n    {cells}\n</div>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{
    width: {W}px;
    height: {H}px;
    overflow: hidden;
    background: #0a0a0a;
}}
{grid_css}

@keyframes kenburns {{
    0%   {{ transform: scale(1.0) translate(0, 0); }}
    100% {{ transform: scale(1.08) translate(-1%, -1%); }}
}}

/* Subtle vignette overlay for cinematic look */
.overlay {{
    position: absolute;
    inset: 0;
    background: radial-gradient(
        ellipse at center,
        transparent 50%,
        rgba(0, 0, 0, 0.4) 100%
    );
    pointer-events: none;
}}
</style>
</head>
<body>
    {grid_html}
    <div class="overlay"></div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Browser-based frame capture (like HyperFrames engine's Puppeteer capture)
# ---------------------------------------------------------------------------

async def _capture_scenes_playwright(
    html_files: list[str], output_pngs: list[str]
) -> None:
    """Capture screenshots of HTML compositions using Playwright.

    Similar to HyperFrames' frameCapture service which uses Puppeteer to
    navigate to HTML pages and capture deterministic screenshots.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--no-sandbox",
            ],
        )
        # Single page, reuse for all scenes (like HyperFrames session reuse)
        page = await browser.new_page(viewport={"width": W, "height": H})

        for html_file, out_png in zip(html_files, output_pngs):
            file_url = Path(html_file).as_uri()
            await page.goto(file_url, wait_until="networkidle")
            # Wait for animations to settle (similar to HyperFrames' seek protocol)
            await page.wait_for_timeout(200)
            await page.screenshot(path=out_png, type="png")

        await browser.close()


def capture_scenes(html_files: list[str], output_pngs: list[str]) -> None:
    """Synchronous wrapper for Playwright scene capture."""
    asyncio.run(_capture_scenes_playwright(html_files, output_pngs))


# ---------------------------------------------------------------------------
# TTS (same as before)
# ---------------------------------------------------------------------------

def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


async def tts_edge(text: str, voice: str, rate: str, out_wav: str) -> None:
    """edge-tts (online) outputs mp3; convert to wav so we can read duration."""
    import ssl
    import aiohttp

    mp3 = out_wav.replace(".wav", ".mp3")

    ssl_ctx: ssl.SSLContext | bool | None = None
    if os.getenv("NODE_TLS_REJECT_UNAUTHORIZED", "1") == "0":
        ssl_ctx = False

    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None

    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    communicate = edge_tts.Communicate(
        text, voice=voice, rate=rate, proxy=proxy, connector=connector
    )
    await communicate.save(mp3)

    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3, "-ac", "1", "-ar", "24000", out_wav],
        check=True, capture_output=True,
    )
    os.remove(mp3)


def tts_dummy(text: str, out_wav: str, duration: float = 3.0) -> None:
    """Create a silent WAV file for testing (when TTS is unavailable)."""
    import struct

    sample_rate = 24000
    num_samples = int(duration * sample_rate)

    with wave.open(out_wav, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for _ in range(num_samples):
            wav.writeframes(struct.pack("<h", 0))


def tts_offline(text: str, out_wav: str, prefer_lang: str = "zh") -> None:
    """Offline fallback using pyttsx3 (Windows SAPI or Linux espeak)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        chosen = None
        for v in voices:
            blob = f"{v.id} {v.name} {getattr(v, 'languages', '')}".lower()
            if prefer_lang in blob or "chinese" in blob or "zh" in blob:
                chosen = v.id
                break
        if chosen:
            engine.setProperty("voice", chosen)
        engine.save_to_file(text, out_wav)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        raise RuntimeError(
            f"Offline TTS failed: {e}\n"
            "Install espeak: sudo apt-get install espeak espeak-ng (Linux)\n"
            "Or install edge-tts: pip install edge-tts (recommended)"
        )


def synth_line(text: str, voice: str, rate: str, out_wav: str, use_offline: bool, use_dummy: bool = False) -> None:
    """Try online edge-tts first; fall back to offline TTS, then dummy TTS."""
    if use_dummy:
        duration = len(text) * 0.15
        tts_dummy(text, out_wav, duration)
        return

    if not use_offline and edge_tts is not None:
        try:
            asyncio.run(tts_edge(text, voice, rate, out_wav))
            return
        except Exception as e:
            print(f"    [!] edge-tts failed ({type(e).__name__}); trying offline TTS")

    try:
        tts_offline(text, out_wav)
        return
    except Exception as e:
        print(f"    [!] Offline TTS also failed: {type(e).__name__}")

    print(f"    [i] Using dummy TTS (silent audio) - video will have no narration")
    duration = len(text) * 0.15
    tts_dummy(text, out_wav, duration)


# ---------------------------------------------------------------------------
# SRT generation
# ---------------------------------------------------------------------------

def srt_ts(t: float) -> str:
    h, m, s, ms = int(t // 3600), int((t % 3600) // 60), int(t % 60), int((t % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Video generation (HTML composition → browser capture → FFmpeg encode)
# ---------------------------------------------------------------------------

def generate_video(
    lines: list[tuple[str, str | None]],
    out_path: str = "output.mp4",
    srt_path: str = "output.srt",
    voice: str = "zh-TW-YunJheNeural",
    rate: str = "+8%",
    use_offline: bool = False,
    use_images: bool = True,
    use_dummy: bool = False,
) -> None:
    """Core video generation logic — HTML composition + browser capture approach.

    This mirrors the HyperFrames rendering pipeline:
    1. Build HTML compositions (instead of Pillow images)
    2. Capture frames via headless browser (Playwright, like HyperFrames' Puppeteer)
    3. Encode with FFmpeg (image → video segments → concat → mux with audio)
    """
    total = len(lines)
    work = Path(tempfile.mkdtemp(prefix="vid_"))
    print(f"[i] {total} scenes · voice={voice} · images={use_images} · work={work}")

    # Step 1: Synthesize TTS audio per line
    print("[1/4] Synthesizing voiceover...")
    wavs, durations = [], []
    for i, (text, _query) in enumerate(lines):
        wav = str(work / f"a_{i:03d}.wav")
        synth_line(text, voice, rate, wav, use_offline, use_dummy)
        dur = wav_duration(wav)
        wavs.append(wav)
        durations.append(dur)
        print(f"    [{i+1:>3}/{total}] {dur:5.1f}s  {text[:30]}")

    # Step 2: Fetch background images (optional)
    print("[2/4] Preparing background assets...")
    bg_images: list[list[str]] = []
    for i, (text, query) in enumerate(lines):
        paths: list[str] = []
        used_q = ""
        if use_images:
            paths, used_q = fetch_images(text, query, width=W, count=6)
            paths = [p for p in paths if os.path.exists(p)]
        bg_images.append(paths)
        tag = f"img='{used_q}'({len(paths)})" if used_q else "img=none"
        print(f"    [{i+1:>3}/{total}] {tag:<32}")

    # Step 3: Generate HTML compositions and capture via browser
    print("[3/4] Rendering HTML compositions via headless browser...")
    html_files, png_files = [], []
    for i, (text, _query) in enumerate(lines):
        html_path = str(work / f"scene_{i:03d}.html")
        png_path = str(work / f"s_{i:03d}.png")
        html_content = _scene_html(text, i, total, bg_images[i])
        Path(html_path).write_text(html_content, encoding="utf-8")
        html_files.append(html_path)
        png_files.append(png_path)

    # Capture all scenes using Playwright (like HyperFrames' frame capture service)
    capture_scenes(html_files, png_files)
    print(f"    Captured {total} frames via Playwright")

    # Step 4: Encode with FFmpeg
    print("[4/4] Encoding video with FFmpeg...")

    # Concat all audio
    audio_list = work / "audio.txt"
    audio_list.write_text("".join(f"file '{w}'\n" for w in wavs), encoding="utf-8")
    full_audio = str(work / "full.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(audio_list), "-c", "copy", full_audio],
        check=True, capture_output=True,
    )

    # Per-scene video segments (image held for its audio duration)
    segs = []
    for i, (png, dur) in enumerate(zip(png_files, durations)):
        seg = str(work / f"v_{i:03d}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loop", "1", "-i", png, "-t", f"{dur:.3f}",
             "-r", str(FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", f"scale={W}:{H}", seg],
            check=True, capture_output=True,
        )
        segs.append(seg)

    # Concatenate video segments
    video_list = work / "video.txt"
    video_list.write_text("".join(f"file '{s}'\n" for s in segs), encoding="utf-8")
    silent_video = str(work / "silent.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", str(video_list), "-c", "copy", silent_video],
        check=True, capture_output=True,
    )

    # Mux audio + video
    subprocess.run(
        ["ffmpeg", "-y", "-i", silent_video, "-i", full_audio,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-shortest", out_path],
        check=True, capture_output=True,
    )

    # Emit SRT subtitles
    t = 0.0
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, ((text, _q), dur) in enumerate(zip(lines, durations), 1):
            f.write(f"{i}\n{srt_ts(t)} --> {srt_ts(t + dur)}\n{text}\n\n")
            t += dur

    print(f"[done] wrote {out_path} ({sum(durations):.1f}s) and {srt_path}")
