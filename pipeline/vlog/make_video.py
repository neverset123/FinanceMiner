"""
End-to-end "context video" generator (faceless AI-narration explainer).

Pipeline
--------
1. Read a plain-text SCRIPT (one paragraph / one on-screen scene per line).
2. Synthesize a voiceover per line with edge-tts (free, natural zh-TW/zh-CN voices),
   capturing the exact spoken duration of each line.
3. Build a matching background image (title card) per line with Pillow.
4. Concatenate the audio, render each image for its line's duration,
   and mux everything into a single MP4 with ffmpeg.
5. Also emit an .srt so the captions are re-usable / editable.

This mirrors the structure of the analyzed video:
    hook -> thesis -> 3 numbered points -> escalation -> CTA.

Usage
-----
     python make_video.py --script script.txt --out output.mp4
     python make_video.py --voice zh-CN-YunxiNeural

Requirements: edge-tts, pillow, ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import tempfile
import wave
from pathlib import Path

try:
    import edge_tts  # optional; only needed for online TTS
except Exception:  # pragma: no cover
    edge_tts = None
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    from fetch_images import fetch_image  # web photo per scene (Wikimedia Commons)
except Exception:  # pragma: no cover
    fetch_image = None

W, H = 1280, 720
BG = (17, 24, 39)      # dark slate
FG = (243, 244, 246)   # near white
ACCENT = (250, 204, 21)  # amber

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


def load_font(size: int) -> ImageFont.FreeTypeFont:
    # Try common CJK-capable fonts across platforms; fall back to default.
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",       # Microsoft YaHei
        r"C:\Windows\Fonts\msjh.ttc",       # Microsoft JhengHei
        r"C:\Windows\Fonts\simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_cjk(draw, text, font, max_width) -> list[str]:
    """Greedy character-based wrapping (works for CJK where there are no spaces)."""
    lines, cur = [], ""
    for ch in text:
        test = cur + ch
        if draw.textlength(test, font=font) <= max_width or not cur:
            cur = test
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def cover_resize(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resize + center-crop an image to exactly (w, h), preserving aspect ratio."""
    src_ratio = img.width / img.height
    dst_ratio = w / h
    if src_ratio > dst_ratio:
        new_h = h
        new_w = int(h * src_ratio)
    else:
        new_w = w
        new_h = int(w / src_ratio)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return img.crop((left, top, left + w, top + h))


def make_slide(text: str, index: int, total: int, path: str,
               query: str | None = None, use_images: bool = True) -> str:
    """Build one scene image. Returns the image-search query used (for logging)."""
    used_query = ""
    bg_img = None
    if use_images and fetch_image is not None:
        img_path, used_query = fetch_image(text, query, width=W)
        if img_path and os.path.exists(img_path):
            try:
                bg_img = cover_resize(Image.open(img_path).convert("RGB"), W, H)
            except Exception:
                bg_img = None

    if bg_img is None:
        img = Image.new("RGB", (W, H), BG)
    else:
        img = bg_img

    img.save(path)
    return used_query


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


async def tts_edge(text: str, voice: str, rate: str, out_wav: str) -> None:
    """edge-tts (online) outputs mp3; convert to wav so we can read duration."""
    import ssl
    import aiohttp

    mp3 = out_wav.replace(".wav", ".mp3")

    # Respect NODE_TLS_REJECT_UNAUTHORIZED=0 (proxy with SSL interception)
    ssl_ctx: ssl.SSLContext | bool | None = None
    if os.getenv("NODE_TLS_REJECT_UNAUTHORIZED", "1") == "0":
        ssl_ctx = False  # disable SSL verification

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
    import wave
    import struct
    
    sample_rate = 24000
    num_samples = int(duration * sample_rate)
    
    with wave.open(out_wav, 'w') as wav:
        wav.setnchannels(1)  # mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        # Write silence (zeros)
        for _ in range(num_samples):
            wav.writeframes(struct.pack('<h', 0))


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
    # Use dummy TTS if explicitly requested
    if use_dummy:
        duration = len(text) * 0.15  # ~150ms per character (rough estimate for Chinese)
        tts_dummy(text, out_wav, duration)
        return
    
    # Try online edge-tts
    if not use_offline and edge_tts is not None:
        try:
            asyncio.run(tts_edge(text, voice, rate, out_wav))
            return
        except Exception as e:  # network/SSL/proxy blocked
            print(f"    [!] edge-tts failed ({type(e).__name__}); trying offline TTS")
    
    # Try offline TTS (pyttsx3 + espeak)
    try:
        tts_offline(text, out_wav)
        return
    except Exception as e:
        print(f"    [!] Offline TTS also failed: {type(e).__name__}")
    
    # Last resort: use dummy/silent audio
    print(f"    [i] Using dummy TTS (silent audio) - video will have no narration")
    duration = len(text) * 0.15
    tts_dummy(text, out_wav, duration)


def srt_ts(t: float) -> str:
    h, m, s, ms = int(t // 3600), int((t % 3600) // 60), int(t % 60), int((t % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

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
    """Core video generation logic — callable programmatically.

    Args:
        lines: List of (spoken_text, image_query_or_None) tuples.
        out_path: Output MP4 file path.
        srt_path: Output SRT subtitle path.
        voice: edge-tts voice name.
        rate: Speech rate string (e.g. "+8%").
        use_offline: If True, skip edge-tts and use offline SAPI.
        use_images: If True, fetch web photos for backgrounds.
        use_dummy: If True, use silent audio (for testing without TTS).
    """
    total = len(lines)
    work = Path(tempfile.mkdtemp(prefix="vid_"))
    print(f"[i] {total} scenes · voice={voice} · images={use_images} · work={work}")

    wavs, imgs, durations = [], [], []
    for i, (text, query) in enumerate(lines):
        wav = str(work / f"a_{i:03d}.wav")
        img = str(work / f"s_{i:03d}.png")
        synth_line(text, voice, rate, wav, use_offline, use_dummy)
        dur = wav_duration(wav)
        used_q = make_slide(text, i, total, img, query=query, use_images=use_images)
        wavs.append(wav); imgs.append(img); durations.append(dur)
        tag = f"img='{used_q}'" if used_q else "img=none"
        print(f"    [{i+1:>3}/{total}] {dur:5.1f}s  {tag:<28} {text[:20]}")

    # concat audio
    audio_list = work / "audio.txt"
    audio_list.write_text("".join(f"file '{w}'\n" for w in wavs), encoding="utf-8")
    full_audio = str(work / "full.wav")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(audio_list), "-c", "copy", full_audio],
                   check=True, capture_output=True)

    # per-image video segments (image shown for its line's audio duration)
    segs = []
    for i, (img, dur) in enumerate(zip(imgs, durations)):
        seg = str(work / f"v_{i:03d}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", img, "-t", f"{dur:.3f}",
                        "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-vf", f"scale={W}:{H}", seg],
                       check=True, capture_output=True)
        segs.append(seg)

    video_list = work / "video.txt"
    video_list.write_text("".join(f"file '{s}'\n" for s in segs), encoding="utf-8")
    silent_video = str(work / "silent.mp4")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(video_list), "-c", "copy", silent_video],
                   check=True, capture_output=True)

    # mux audio + video
    subprocess.run(["ffmpeg", "-y", "-i", silent_video, "-i", full_audio,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", out_path],
                   check=True, capture_output=True)

    # emit srt
    t = 0.0
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, ((text, _q), dur) in enumerate(zip(lines, durations), 1):
            f.write(f"{i}\n{srt_ts(t)} --> {srt_ts(t + dur)}\n{text}\n\n")
            t += dur

    print(f"[✓] wrote {out_path} ({sum(durations):.1f}s) and {srt_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="script.txt")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--voice", default="zh-TW-YunJheNeural",
                    help="edge-tts voice, e.g. zh-TW-YunJheNeural, zh-TW-HsiaoChenNeural, zh-CN-YunxiNeural")
    ap.add_argument("--rate", default="+8%", help="speech rate, e.g. +0%, +10%, -5%")
    ap.add_argument("--srt", default="output.srt")
    ap.add_argument("--offline", action="store_true",
                    help="skip online edge-tts and use offline Windows SAPI (pyttsx3)")
    ap.add_argument("--no-images", action="store_true",
                    help="disable web photo backgrounds (use plain title cards)")
    ap.add_argument("--dummy-tts", action="store_true",
                    help="use silent audio (for testing image selection without TTS)")
    args = ap.parse_args()

    lines = read_script(args.script)
    generate_video(
        lines=lines,
        out_path=args.out,
        srt_path=args.srt,
        voice=args.voice,
        rate=args.rate,
        use_offline=args.offline,
        use_images=not args.no_images,
        use_dummy=args.dummy_tts,
    )


if __name__ == "__main__":
    main()
