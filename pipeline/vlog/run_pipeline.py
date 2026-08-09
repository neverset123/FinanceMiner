#!/usr/bin/env python3
"""
Single orchestrator for the vlog pipeline.

Runs the full pipeline end-to-end:
  1. Parse script.txt
  2. Extract English image-search keywords via OpenAI LLM
  3. Fetch images from Wikimedia Commons
  4. Generate TTS + slides + mux into final MP4

Usage:
    python run_pipeline.py --script script.txt --out output.mp4
    python run_pipeline.py --script script.txt --out output.mp4 --voice zh-CN-YunxiNeural
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local imports work when run from any directory
sys.path.insert(0, str(Path(__file__).parent))

from extract_keywords import extract_keywords
from make_video import generate_video, read_script


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vlog pipeline: script → LLM keywords → images → TTS → MP4"
    )
    ap.add_argument("--script", default="script.txt", help="Path to script.txt")
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
        help="Disable web photo backgrounds (plain title cards only)",
    )
    args = ap.parse_args()

    if args.srt is None:
        args.srt = str(Path(args.out).with_suffix(".srt"))

    # Step 1: Read script
    lines = read_script(args.script)
    texts = [text for text, _q in lines]
    explicit_queries = [q for _text, q in lines]
    print(f"[1/3] Parsed {len(lines)} lines from {args.script}")

    # Step 2: Extract keywords via LLM (skip lines that already have @@ overrides)
    print("[2/3] Extracting keywords via OpenAI LLM...")
    lines_needing_keywords = [
        (i, texts[i]) for i in range(len(texts)) if explicit_queries[i] is None
    ]

    if lines_needing_keywords:
        indices, bare_texts = zip(*lines_needing_keywords)
        llm_keywords = extract_keywords(list(bare_texts))

        # Merge: use @@ override where present, LLM keyword otherwise
        queries: list[str] = []
        llm_idx = 0
        for i in range(len(texts)):
            if explicit_queries[i] is not None:
                queries.append(explicit_queries[i])
            else:
                queries.append(llm_keywords[llm_idx])
                llm_idx += 1
    else:
        queries = [q or "global finance economy" for q in explicit_queries]

    for i, (text, q) in enumerate(zip(texts, queries), 1):
        print(f"    [{i:>2}] {text[:25]:25s} → {q}")

    # Step 3: Generate video
    print("[3/3] Generating video (TTS + slides + mux)...")
    generate_video(
        lines=list(zip(texts, queries)),
        out_path=args.out,
        srt_path=args.srt,
        voice=args.voice,
        rate=args.rate,
        use_offline=args.offline,
        use_images=not args.no_images,
    )


if __name__ == "__main__":
    main()
