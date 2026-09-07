"""Shared helpers for the Route A YouTube -> TeleMem transcript pipeline.

Route A ("transcript-only") does not download or decode any video. It:
  1. Enumerates a channel's uploads with yt-dlp (flat, no media download).
  2. Downloads each video's captions (manual or auto) as VTT/SRT.
  3. Parses the captions into clean plain text.
  4. Stores the text in TeleMem so an LLM can semantically search it later.

Only text is fetched, so this runs without a GPU or a VLM endpoint. It still
needs an LLM + embedder endpoint if you ingest with ``infer=True`` (the default),
because TeleMem uses the LLM to extract structured facts from each transcript.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse, parse_qs


# --------------------------------------------------------------------------- #
#                              Channel enumeration                            #
# --------------------------------------------------------------------------- #
def normalize_channel_url(channel: str) -> str:
    """Accept a handle, channel id, or full URL and return a /videos listing URL."""
    channel = channel.strip()
    if channel.startswith(("http://", "https://")):
        # Point plain channel URLs at the uploads tab for a complete listing.
        parsed = urlparse(channel)
        path = parsed.path.rstrip("/")
        if any(path.endswith(seg) for seg in ("/videos", "/streams", "/shorts")):
            return channel
        if "/playlist" in path or parsed.path.startswith("/watch"):
            return channel
        return f"{channel.rstrip('/')}/videos"
    if channel.startswith("@"):
        return f"https://www.youtube.com/{channel}/videos"
    if channel.startswith(("UC", "UU")) and len(channel) >= 20:
        return f"https://www.youtube.com/channel/{channel}/videos"
    # Fall back to treating it as a handle.
    return f"https://www.youtube.com/@{channel}/videos"


def list_channel_videos(channel: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Return ``[{"id", "url", "title"}]`` for a channel without downloading media."""
    import yt_dlp

    listing_url = normalize_channel_url(channel)
    ydl_opts: Dict[str, Any] = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": True,
        "ignoreerrors": True,
    }
    if limit:
        ydl_opts["playlistend"] = limit

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(listing_url, download=False)

    entries = _flatten_entries(info)
    videos: List[Dict[str, str]] = []
    seen: set[str] = set()
    for entry in entries:
        if not entry:
            continue
        vid = entry.get("id")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        videos.append(
            {
                "id": vid,
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
                "title": entry.get("title") or "",
                "upload_date": entry.get("upload_date") or "",
            }
        )
        if limit and len(videos) >= limit:
            break
    return videos


def _flatten_entries(info: Optional[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    """Yield leaf video entries from a possibly nested yt-dlp playlist result."""
    if not info:
        return
    entries = info.get("entries")
    if entries is None:
        yield info
        return
    for entry in entries:
        if entry and entry.get("entries") is not None:
            yield from _flatten_entries(entry)
        elif entry:
            yield entry


# --------------------------------------------------------------------------- #
#                             Transcript download                             #
# --------------------------------------------------------------------------- #
def download_transcript(
    video_url: str,
    dest_dir: str,
    langs: Optional[List[str]] = None,
) -> Optional[str]:
    """Download captions for one video. Returns the caption file path or None.

    Prefers manual subtitles, falls back to auto-generated. Requests VTT which
    yt-dlp can write natively (no ffmpeg needed).
    """
    import yt_dlp

    langs = langs or ["en"]
    os.makedirs(dest_dir, exist_ok=True)
    video_id = _extract_video_id(video_url)

    with tempfile.TemporaryDirectory() as tmp:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": langs,
            "subtitlesformat": "vtt",
            "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
            "quiet": True,
            "ignoreerrors": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
        if info and info.get("id"):
            video_id = info["id"]

        picked = _pick_subtitle_file(tmp, video_id, langs)
        if not picked:
            return None

        ext = os.path.splitext(picked)[1]
        final_path = os.path.join(dest_dir, f"{video_id}{ext}")
        Path(picked).replace(final_path)
        return final_path


def _pick_subtitle_file(folder: str, video_id: str, langs: List[str]) -> Optional[str]:
    """Choose the best caption file, preferring requested languages in order."""
    candidates = [
        f for f in os.listdir(folder)
        if f.startswith(video_id) and f.endswith((".vtt", ".srt"))
    ]
    if not candidates:
        return None
    for lang in langs:
        for f in candidates:
            if f".{lang}." in f or f".{lang}-" in f:
                return os.path.join(folder, f)
    return os.path.join(folder, candidates[0])


# --------------------------------------------------------------------------- #
#                             Transcript parsing                              #
# --------------------------------------------------------------------------- #
_TS_LINE = re.compile(r"-->")
_VTT_TAG = re.compile(r"<[^>]+>")
_CUE_SETTING = re.compile(r"\balign:|\bposition:|\bsize:|\bline:")


def transcribe_with_whisper(
    video_url: str,
    dest_dir: str,
    model_name: str = "medium",
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
) -> Optional[str]:
    """Download audio and transcribe with faster-whisper.

    Returns the path to a ``.txt`` file containing the transcript, or ``None``
    if the download or transcription failed.

    The model is loaded fresh each call (fine for one-shot CLI runs).

    Env vars:
      WHISPER_DEVICE     – ``auto`` (default), ``cpu``, or ``cuda``
      WHISPER_MODEL_PATH – local directory containing the CTranslate2 model files;
                           when set, ``model_name`` is ignored and no download occurs.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is required for audio transcription. "
            "Install it with: pip install faster-whisper"
        ) from exc

    import yt_dlp

    os.makedirs(dest_dir, exist_ok=True)
    video_id = _extract_video_id(video_url)

    with tempfile.TemporaryDirectory() as tmp:
        print(f"    [whisper] downloading audio to {tmp} …", flush=True)
        ydl_opts = {
            # Prefer m4a so faster-whisper can read it without a separate ffmpeg step.
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            "outtmpl": os.path.join(tmp, f"{video_id}.%(ext)s"),
            "quiet": False,
            "ignoreerrors": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
        if not info:
            print("    [whisper] audio download returned no info", flush=True)
            return None

        audio_files = [
            f for f in os.listdir(tmp)
            if f.startswith(video_id) and os.path.splitext(f)[1] in {".m4a", ".mp3", ".webm", ".opus", ".ogg"}
        ]
        if not audio_files:
            print("    [whisper] no audio file found after download", flush=True)
            return None
        audio_path = os.path.join(tmp, audio_files[0])
        print(f"    [whisper] audio ready: {audio_files[0]} ({os.path.getsize(audio_path) // 1024}KB)", flush=True)

        device = os.getenv("WHISPER_DEVICE", "auto")
        compute = "float16" if device == "cuda" else "int8"
        model_path = os.getenv("WHISPER_MODEL_PATH") or model_name
        print(f"    [whisper] loading model from {model_path!r} (device={device}) …", flush=True)
        model = WhisperModel(model_path, device=device, compute_type=compute)
        print(f"    [whisper] transcribing …", flush=True)
        # Use initial_prompt to nudge Whisper toward simplified Chinese output
        # when the audio is Chinese. Whisper's Chinese output style is influenced
        # by the prompt: providing simplified Chinese text biases decoding accordingly.
        transcribe_lang = language
        if initial_prompt is None:
            if (language or "").startswith("zh"):
                initial_prompt = "以下是普通话的句子。"
            elif language is None:
                # Auto-detect: run a short detection pass first so we can apply
                # the simplified-Chinese prompt when needed.
                _, detect_info = model.transcribe(audio_path, language=None, beam_size=1)
                detected = detect_info.language
                print(f"    [whisper] detected language: {detected} (p={detect_info.language_probability:.2f})", flush=True)
                if detected == "zh":
                    initial_prompt = "以下是普通话的句子。"
                    transcribe_lang = "zh"
        segments, info2 = model.transcribe(
            audio_path, language=transcribe_lang, beam_size=5,
            initial_prompt=initial_prompt,
        )
        print(f"    [whisper] transcription language: {info2.language} (p={info2.language_probability:.2f})", flush=True)

        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        if not text:
            print("    [whisper] transcription produced empty text", flush=True)
            return None

        out_path = os.path.join(dest_dir, f"{video_id}.txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"    [whisper] saved transcript: {out_path} ({len(text)} chars)", flush=True)
        return out_path


def transcript_to_text(path: str) -> str:
    """Parse a VTT/SRT caption file or plain .txt into clean plain text."""
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read().strip()

    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw_lines = [ln.rstrip("\n") for ln in fh]

    text_lines: List[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in ("WEBVTT",) or stripped.startswith(("NOTE", "Kind:", "Language:")):
            continue
        if _TS_LINE.search(stripped) or _CUE_SETTING.search(stripped):
            continue
        if stripped.isdigit():  # SRT sequence index
            continue
        cleaned = _VTT_TAG.sub("", stripped).strip()
        if not cleaned:
            continue
        # YouTube auto-captions repeat rolling lines; drop consecutive dupes.
        if text_lines and text_lines[-1] == cleaned:
            continue
        text_lines.append(cleaned)

    return _dedupe_overlap(text_lines)


def _dedupe_overlap(lines: List[str]) -> str:
    """Join lines, removing the rolling-window repetition of auto-captions."""
    out: List[str] = []
    for line in lines:
        if out and line in out[-1]:
            continue
        if out and out[-1] in line:
            out[-1] = line
            continue
        out.append(line)
    return " ".join(out).strip()


def _extract_video_id(video_url: str) -> str:
    parsed = urlparse(video_url)
    if parsed.netloc.endswith("youtu.be"):
        return parsed.path.lstrip("/")
    qs = parse_qs(parsed.query)
    if "v" in qs:
        return qs["v"][0]
    return video_url.rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- #
#                        Simple processed-id state file                       #
# --------------------------------------------------------------------------- #
def load_state(path: str) -> Dict[str, Any]:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"processed": []}


def save_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)


def chunk_text(text: str, chunk_chars: int, overlap: int = 200) -> List[str]:
    """Split long transcripts into overlapping character windows for ingestion."""
    if chunk_chars <= 0 or len(text) <= chunk_chars:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start = end - overlap
    return chunks
