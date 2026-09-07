"""Ingest a whole YouTube channel's transcripts into TeleMem memory.

Usage examples
--------------
    # ingest the 20 most recent uploads of a channel into TeleMem
    python ingest_channel.py --channel "@Bloomberg" --limit 20 --user-id Bloomberg

    # use a specific TeleMem config (LLM/embedder endpoints) and store raw text
        python ingest_channel.py --channel https://www.youtube.com/@Bloomberg \
        --user-id Bloomberg --no-infer

Notes
-----
* ``--infer`` (default) sends each transcript chunk through the LLM so TeleMem
  extracts structured facts. This needs a working ``llm`` + ``embedder`` in your
  TeleMem config. ``--no-infer`` stores the raw transcript text verbatim and only
  needs the ``embedder``.
* Already-processed video ids are recorded in the state file and skipped on
  re-runs, so this is safe to run repeatedly (e.g. from cron).
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import yt_utils

from config import make_memory as _make_memory, build_telemem_config


def _ensure_upload_date_column(db_path: str) -> None:
    """Add the ``upload_date`` column to history if it doesn't exist yet."""
    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(history)")}
        if "upload_date" not in cols:
            conn.execute("ALTER TABLE history ADD COLUMN upload_date TEXT")
            conn.commit()
    finally:
        conn.close()


def _set_upload_date(db_path: str, memory_ids: list[str], upload_date: str) -> None:
    """Stamp ``upload_date`` on history rows matching *memory_ids*."""
    if not memory_ids or not upload_date:
        return
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ",".join("?" for _ in memory_ids)
        conn.execute(
            f"UPDATE history SET upload_date = ? "
            f"WHERE memory_id IN ({placeholders}) AND upload_date IS NULL",
            [upload_date, *memory_ids],
        )
        conn.commit()
    finally:
        conn.close()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", required=True, help="Channel handle (@name), channel id (UC...), or URL")
    p.add_argument("--user-id", required=True, help="TeleMem user_id to namespace this channel's memories")
    p.add_argument("--limit", type=int, default=None, help="Max number of videos to ingest (newest first)")
    p.add_argument("--langs", default="en", help="Comma-separated caption languages to try, in order")
    p.add_argument("--config", default=os.getenv("TELEMEM_CONFIG"), help="Path to a TeleMem config YAML")
    p.add_argument("--work-dir", default=str(Path(__file__).parent / "data"), help="Where to keep transcripts/state")
    p.add_argument("--chunk-chars", type=int, default=4000, help="Max characters per memory chunk (0 = no split)")
    p.add_argument("--infer", dest="infer", action="store_true", default=True, help="LLM fact extraction (default)")
    p.add_argument("--no-infer", dest="infer", action="store_false", help="Store raw transcript text verbatim")
    p.add_argument("--keep-transcripts", action="store_true", help="Do not delete caption files after ingest")
    p.add_argument(
        "--whisper-model",
        default=None,
        metavar="SIZE",
        help="Fall back to faster-whisper transcription when captions are unavailable. "
             "SIZE is the model name: tiny, base, small, medium (default), large-v3. "
             "Requires: pip install faster-whisper",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    work = Path(args.work_dir)
    transcripts_dir = work / "transcripts"
    state_path = str(work / f"state_{args.user_id}.json")
    state = yt_utils.load_state(state_path)
    processed = set(state.get("processed", []))

    print(f"Enumerating channel: {args.channel}")
    videos = yt_utils.list_channel_videos(args.channel, limit=args.limit)
    print(f"Found {len(videos)} videos ({len(processed)} already processed).")
    if not videos:
        print("Nothing to do.")
        return 0

    memory = _make_memory(args.config)

    # Resolve history.db path and ensure upload_date column exists
    cfg = build_telemem_config()
    history_db = cfg.get("history_db_path", "db/history.db")
    _ensure_upload_date_column(history_db)

    added, skipped, failed = 0, 0, 0
    for i, video in enumerate(videos, 1):
        vid = video["id"]
        title = video["title"] or vid
        if vid in processed:
            skipped += 1
            continue

        print(f"[{i}/{len(videos)}] {vid} :: {title}")
        try:
            sub_path = yt_utils.download_transcript(video["url"], str(transcripts_dir), langs=langs)
            if not sub_path:
                if args.whisper_model:
                    print(f"    no captions, transcribing with whisper ({args.whisper_model})…")
                    sub_path = yt_utils.transcribe_with_whisper(
                        video["url"], str(transcripts_dir), model_name=args.whisper_model
                    )
                if not sub_path:
                    print("    no captions available, skipping")
                    failed += 1
                    continue

            text = yt_utils.transcript_to_text(sub_path)
            if not args.keep_transcripts:
                try:
                    os.remove(sub_path)
                except OSError:
                    pass

            if not text:
                print("    empty transcript, skipping")
                failed += 1
                continue

            chunks = yt_utils.chunk_text(text, args.chunk_chars)
            upload_date = video.get("upload_date") or ""
            memory_ids = []
            for j, chunk in enumerate(chunks):
                metadata = {
                    "source": "youtube",
                    "video_id": vid,
                    "title": title,
                    "url": video["url"],
                    "upload_date": upload_date,
                    "chunk": j,
                    "chunks_total": len(chunks),
                }
                result = memory.add(chunk, user_id=args.user_id, metadata=metadata, infer=args.infer)
                # Collect memory IDs from the result to backfill upload_date in history.db
                if result and isinstance(result, dict):
                    for r in result.get("results", []):
                        mid = r.get("id") or r.get("memory_id")
                        if mid:
                            memory_ids.append(mid)

            _set_upload_date(history_db, memory_ids, upload_date)

            processed.add(vid)
            state["processed"] = sorted(processed)
            yt_utils.save_state(state_path, state)
            added += 1
            print(f"    stored {len(chunks)} chunk(s)")
        except KeyboardInterrupt:
            print("\nInterrupted; progress saved.")
            break
        except Exception as exc:  # keep going on a single bad video
            import traceback
            failed += 1
            print(f"    ERROR: {exc}")
            traceback.print_exc()

    print(f"\nDone. added={added} skipped={skipped} failed={failed}")
    print(f"State: {state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
