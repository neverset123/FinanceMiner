#!/usr/bin/env python3
"""
Orchestrator for the vlog pipeline.

Steps:
  1. Read transcript.txt and split into scenes via split_into_chunks (handles
     both single-line and multi-line transcripts).
  2. Extract English image-search keywords via LLM for each scene
     (uses extract_keywords.py).
  3. Generate the video: fetch Wikimedia images, render HTML scenes via headless
     Chromium, synthesise voiceover, encode to MP4, write SRT (make_video.py).

Usage:
    python run_pipeline.py --script transcript.txt --out output.mp4
    python run_pipeline.py --script transcript.txt --out output.mp4 --voice zh-CN-YunxiNeural
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extract_keywords import extract_keywords, split_into_chunks
from make_video import generate_video


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vlog pipeline: script → LLM keywords → images → HTML → Playwright → MP4"
    )
    ap.add_argument("--script", default="transcript.txt", help="Path to script file")
    ap.add_argument("--out", default="output.mp4", help="Output MP4 path")
    ap.add_argument(
        "--voice",
        default="zh-TW-YunJheNeural",
        help="edge-tts voice (e.g. zh-TW-YunJheNeural, zh-CN-YunxiNeural)",
    )
    ap.add_argument("--rate", default="+8%", help="Speech rate (e.g. +0%%, +10%%, -5%%)")
    ap.add_argument("--srt", default=None, help="SRT output path (default: <out>.srt)")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Skip online edge-tts, use offline Windows SAPI (pyttsx3)",
    )
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Disable web photo backgrounds (plain dark slides only)",
    )
    ap.add_argument(
        "--dummy-tts",
        action="store_true",
        help="Use silent audio (for testing rendering without TTS)",
    )
    args = ap.parse_args()

    if args.srt is None:
        args.srt = str(Path(args.out).with_suffix(".srt"))

    # Step 1: Read and chunk transcript
    texts = split_into_chunks(Path(args.script).read_text(encoding="utf-8"))
    if not texts:
        raise SystemExit(f"No usable lines found in {args.script}")
    print(f"[1/3] Parsed {len(texts)} lines from {args.script}")

    # Step 2: Extract keywords via LLM
    print("[2/3] Extracting keywords via LLM...")
    queries = extract_keywords(texts)

    for i, (text, q) in enumerate(zip(texts, queries), 1):
        print(f"    [{i:>2}] {text[:25]:25s} → {q}")

    # Step 3: Generate video
    print("[3/3] Generating video (HTML → Playwright → FFmpeg)...")
    generate_video(
        lines=list(zip(texts, queries)),
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
