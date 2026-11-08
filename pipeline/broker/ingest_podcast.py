#!/usr/bin/env python3
"""Transcribe podcast episodes and ingest them into TeleMem.

For every episode on a podcast page this downloads the audio, transcribes it
with faster-whisper, chunks the text, and stores it in TeleMem. Downloaded
audio is downloaded to a temp file, transcribed, then deleted immediately.

DB isolation
------------
To guarantee the podcast memories never mix with the broker/YouTube memories,
it uses a DEDICATED vector-store collection *and* dedicated on-disk paths (its
own FAISS collection + its own history DB), in addition to a distinct
``--user-id``. Defaults:
    collection : xiaoyuzhou           (broker default: telemem)
    vector dir : db/faiss_db_podcast  (broker default: db/faiss_db)
    history db : db/history_podcast.db (broker default: db/history.db)

Usage
-----
    python pipeline/broker/ingest_podcast.py --whisper-model medium --user-id crossing
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import requests

# Shared broker helpers (config + transcript utilities) live alongside this file.
_BROKER_DIR = Path(__file__).resolve().parent
if str(_BROKER_DIR) not in sys.path:
    sys.path.insert(0, str(_BROKER_DIR))

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
EPISODE_ID_RE = re.compile(r"/episode/([a-f0-9]{24})")
OG_AUDIO_RE = re.compile(r'<meta[^>]*property="og:audio"[^>]*content="([^"]+)"')
OG_TITLE_RE = re.compile(r'<meta[^>]*property="og:title"[^>]*content="([^"]+)"')
MEDIA_RE = re.compile(r"https://media\.xyzcdn\.net/[^\"\\]+\.(?:m4a|mp3)")
INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def get_html(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def safe_name(name: str, max_len: int = 120) -> str:
    cleaned = INVALID_FS_CHARS.sub("_", name).strip()
    return cleaned[:max_len] if cleaned else "episode"


def extract_episode_ids(html: str) -> list[str]:
    seen: dict[str, None] = {}
    for m in EPISODE_ID_RE.finditer(html):
        seen.setdefault(m.group(1), None)
    return list(seen)


def extract_audio_url(html: str) -> str | None:
    m = OG_AUDIO_RE.search(html)
    if m:
        return m.group(1)
    m = MEDIA_RE.search(html)
    return m.group(0) if m else None


def download_file(session: requests.Session, url: str, dest: Path) -> None:
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                if chunk:
                    fh.write(chunk)
        tmp.replace(dest)


def transcribe_audio_diarized(audio_path: str, language: str | None = None) -> str:
    """Transcribe with speaker diarization using FunASR (ASR + CAM++ in one pass).

    FunASR's SenseVoiceSmall handles ASR and its CAM++ model (trained on 200k
    Mandarin speakers) handles diarization — no HuggingFace token required.
    Models are downloaded from ModelScope on first use.

    Requirements: ``pip install funasr modelscope``
    """
    try:
        from funasr import AutoModel as FunASRAutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError as exc:
        raise ImportError(
            "funasr is required for transcription. "
            "Install with: pip install funasr modelscope"
        ) from exc

    from yt_utils import _to_simplified

    device = "cuda" if os.getenv("WHISPER_DEVICE", "auto") == "cuda" else "cpu"
    print(f"  Loading FunASR (SenseVoiceSmall + CAM++, device={device}) ...")
    funasr_model = FunASRAutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        spk_model="cam++",
        device=device,
        disable_update=True,
    )

    print("  Transcribing with speaker diarization ...")
    result = funasr_model.generate(
        input=audio_path,
        cache={},
        language=language or "auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
        spk_num=2,          # 2-speaker interview: forces correct clustering
    )

    sentences = result[0].get("sentence_info") if result else None
    if not sentences:
        raw = result[0].get("text", "") if result else ""
        return _to_simplified(rich_transcription_postprocess(raw))

    lines: list[str] = []
    current_speaker: str | None = None
    current_texts: list[str] = []

    for s in sentences:
        speaker = f"SPEAKER_{s['spk']}"
        text = rich_transcription_postprocess(s.get("text", "")).strip()
        if not text:
            continue
        if speaker != current_speaker:
            if current_texts and current_speaker is not None:
                lines.append(f"[{current_speaker}]: {' '.join(current_texts)}")
            current_speaker = speaker
            current_texts = [text]
        else:
            current_texts.append(text)

    if current_texts and current_speaker is not None:
        lines.append(f"[{current_speaker}]: {' '.join(current_texts)}")

    return _to_simplified("\n".join(lines))


def build_isolated_memory(collection: str, vector_path: str, history_db: str):
    """Create a TeleMem instance whose storage is fully separate from the broker's.

    Overrides the vector-store collection/path and history DB via env vars that
    ``config.build_telemem_config`` reads, then delegates to the shared
    ``config.make_memory`` so LLM/embedder/SSL handling stays identical.
    """
    import os

    os.environ["VECTOR_STORE_COLLECTION"] = collection
    os.environ["VECTOR_STORE_PATH"] = vector_path
    os.environ["HISTORY_DB_PATH"] = history_db

    from config import make_memory

    return make_memory(None)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--podcast-url",
        default="https://www.xiaoyuzhoufm.com/podcast/60502e253c92d4f62c2a9577",
    )
    p.add_argument("--delay", type=float, default=1.0, help="Seconds between requests")

    p.add_argument("--language", default=None, help="Force transcription language (e.g. zh); auto-detect if unset")
    p.add_argument("--chunk-chars", type=int, default=4000, help="Max characters per memory chunk (0 = no split)")
    p.add_argument("--no-infer", dest="infer", action="store_false", default=True, help="Store raw transcript text verbatim instead of LLM fact extraction")

    # --- Isolation knobs: keep podcast memories separate from broker memories ---
    p.add_argument("--user-id", default="xiaoyuzhou", help="TeleMem user_id namespace for these episodes")
    p.add_argument("--collection", default="xiaoyuzhou", help="Dedicated vector-store collection name")
    p.add_argument("--vector-path", default="db/faiss_db_podcast", help="Dedicated FAISS directory")
    p.add_argument("--history-db", default="db/history_podcast.db", help="Dedicated history DB path")

    p.add_argument("--work-dir", default="./data/podcast_audio", help="Where the processed-state file is kept")


    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    import yt_utils

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    work = Path(args.work_dir).expanduser().resolve()
    state_path = str(work / f"state_{args.user_id}.json")
    state = yt_utils.load_state(state_path)
    processed = set(state.get("processed", []))

    memory = build_isolated_memory(args.collection, args.vector_path, args.history_db)
    print(
        f"Ingesting into collection={args.collection!r} vector_path={args.vector_path!r} "
        f"history_db={args.history_db!r} user_id={args.user_id!r}"
    )

    print("Fetching podcast page ...")
    pod_html = get_html(session, args.podcast_url)
    episode_ids = extract_episode_ids(pod_html)
    print(f"Found {len(episode_ids)} unique episode(s).")

    for i, ep_id in enumerate(episode_ids, start=1):
        ep_url = f"https://www.xiaoyuzhoufm.com/episode/{ep_id}"
        print(f"[{i}/{len(episode_ids)}] {ep_url}")

        if ep_id in processed:
            print("  Already ingested, skipping.")
            continue

        try:
            ep_html = get_html(session, ep_url)
            audio_url = extract_audio_url(ep_html)
            if not audio_url:
                print("  No audio URL found; skipping.")
                continue

            title_m = OG_TITLE_RE.search(ep_html)
            title = title_m.group(1) if title_m else ep_id
            ext = Path(audio_url.split("?")[0]).suffix or ".m4a"

            fd, tmp_name = tempfile.mkstemp(prefix=f"{ep_id}_", suffix=ext)
            os.close(fd)
            dest = Path(tmp_name)

            try:
                print(f"  Downloading ...")
                download_file(session, audio_url, dest)
                size_mb = dest.stat().st_size / (1024 * 1024)
                print(f"  Done ({size_mb:.1f} MB)")

                print("  Transcribing ...")
                text = transcribe_audio_diarized(str(dest), language=args.language)
                if not text:
                    print("  Empty transcript; skipping ingest.")
                    continue

                chunks = yt_utils.chunk_text(text, args.chunk_chars)
                for j, chunk in enumerate(chunks):
                    metadata = {
                        "source": "xiaoyuzhou",
                        "episode_id": ep_id,
                        "title": title,
                        "url": ep_url,
                        "chunk": j,
                        "chunks_total": len(chunks),
                    }
                    memory.add(chunk, user_id=args.user_id, metadata=metadata, infer=args.infer)

                processed.add(ep_id)
                state["processed"] = sorted(processed)
                yt_utils.save_state(state_path, state)
                print(f"  Stored {len(chunks)} chunk(s).")
            finally:
                try:
                    dest.unlink()
                except OSError:
                    pass

        except requests.RequestException as exc:
            print(f"  Failed: {exc}", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nInterrupted; progress saved.")
            break

        if args.delay > 0:
            time.sleep(args.delay)

    print("All done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
