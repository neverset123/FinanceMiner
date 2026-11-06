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


def transcribe_audio(model, audio_path: str, language: str | None = None) -> str:
    """Transcribe a local audio file with a preloaded faster-whisper model."""
    from yt_utils import _to_simplified  # traditional -> simplified Chinese

    transcribe_lang = language
    if language is None:
        _, detect = model.transcribe(audio_path, language=None, beam_size=1)
        if detect.language == "zh":
            transcribe_lang = "zh"
    segments, _ = model.transcribe(audio_path, language=transcribe_lang, beam_size=5)
    return _to_simplified(" ".join(s.text.strip() for s in segments if s.text.strip()))


def _get_whisper_segments(model, audio_path: str, language: str | None = None):
    """Run faster-whisper and return (segments_list, detected_language)."""
    transcribe_lang = language
    if language is None:
        _, detect = model.transcribe(audio_path, language=None, beam_size=1)
        if detect.language == "zh":
            transcribe_lang = "zh"
    segments, info = model.transcribe(audio_path, language=transcribe_lang, beam_size=5)
    return list(segments), info.language


def diarize_audio(
    audio_path: str,
    num_speakers: int | None = None,
    min_speakers: int = 1,
    max_speakers: int = 8,
) -> list[dict]:
    """Speaker diarization using resemblyzer embeddings + agglomerative clustering.

    Fully offline — a ~17 MB speaker-encoder is downloaded automatically by
    resemblyzer on first use (no HuggingFace token required).

    Requirements: ``pip install resemblyzer scikit-learn``
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError as exc:
        raise ImportError(
            "resemblyzer is required for offline diarization. "
            "Install with: pip install resemblyzer"
        ) from exc
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import normalize
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for offline diarization. "
            "Install with: pip install scikit-learn"
        ) from exc

    SAMPLE_RATE = 16_000
    WINDOW_SEC = 1.5
    STEP_SEC = 0.75

    print("  [offline-diarize] preprocessing audio ...")
    wav = preprocess_wav(audio_path)  # resamples to 16 kHz mono float32

    window_len = int(WINDOW_SEC * SAMPLE_RATE)
    step_len = int(STEP_SEC * SAMPLE_RATE)

    slices, centers = [], []
    pos = 0
    while pos + window_len <= len(wav):
        slices.append(wav[pos : pos + window_len])
        centers.append((pos + pos + window_len) / 2 / SAMPLE_RATE)
        pos += step_len

    if not slices:
        return []

    print(f"  [offline-diarize] computing embeddings for {len(slices)} windows ...")
    encoder = VoiceEncoder(device="cpu", verbose=False)
    embeds = np.array([encoder.embed_utterance(s) for s in slices])  # (N, 256)
    embeds = normalize(embeds)

    if num_speakers is not None:
        n_clusters = num_speakers
    else:
        n_clusters = _estimate_speaker_count(embeds, min_speakers, max_speakers)

    print(f"  [offline-diarize] clustering into {n_clusters} speaker(s) ...")
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="cosine", linkage="average"
    ).fit_predict(embeds).tolist()

    print(f"  Diarization (offline): {len(set(labels))} unique speaker(s).")
    return _windows_to_segments(centers, labels, WINDOW_SEC / 2)


def _estimate_speaker_count(embeds, min_k: int, max_k: int) -> int:
    """Pick the cluster count that maximises the silhouette score."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    max_k = min(max_k, len(embeds) - 1)
    if min_k >= max_k:
        return max(min_k, 1)

    best_k, best_score = min_k, -1.0
    for k in range(min_k, max_k + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(embeds)
        try:
            score = silhouette_score(embeds, labels, metric="cosine")
        except Exception:
            score = -1.0
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def _windows_to_segments(
    centers: list[float], labels: list[int], half_win: float
) -> list[dict]:
    """Convert per-window speaker labels into contiguous time segments."""
    if not centers:
        return []
    segments: list[dict] = []
    cur_start = centers[0] - half_win
    cur_label = labels[0]
    for t, lab in zip(centers[1:], labels[1:]):
        if lab != cur_label:
            segments.append({
                "start": max(0.0, cur_start),
                "end": t - half_win,
                "speaker": f"SPEAKER_{cur_label:02d}",
            })
            cur_start = t - half_win
            cur_label = lab
    segments.append({
        "start": max(0.0, cur_start),
        "end": centers[-1] + half_win,
        "speaker": f"SPEAKER_{cur_label:02d}",
    })
    return segments


def _assign_speakers(whisper_segments, diar_segments: list[dict]) -> list[dict]:
    """Map each whisper segment to the diarization speaker with most overlap."""
    result = []
    for seg in whisper_segments:
        best_speaker = "Unknown"
        best_overlap = 0.0
        for d in diar_segments:
            overlap = max(0.0, min(seg.end, d["end"]) - max(seg.start, d["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "speaker": best_speaker,
        })
    return result


def _format_diarized_transcript(labeled_segments: list[dict]) -> str:
    """Merge consecutive same-speaker turns and format as '[SPEAKER_X]: text'."""
    from yt_utils import _to_simplified

    lines: list[str] = []
    current_speaker: str | None = None
    current_texts: list[str] = []

    for seg in labeled_segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg["speaker"]
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


def transcribe_audio_diarized(
    model,
    audio_path: str,
    language: str | None = None,
    num_speakers: int | None = None,
    max_speakers: int = 8,
) -> str:
    """Transcribe with per-speaker labels using resemblyzer + faster-whisper.

    Returns a transcript where each speaker's turns are prefixed with
    ``[SPEAKER_00]: ``, ``[SPEAKER_01]: ``, etc.
    """
    print("  Transcribing (with timestamps) ...")
    whisper_segs, detected_lang = _get_whisper_segments(model, audio_path, language)
    print(f"  Whisper: {len(whisper_segs)} segment(s), lang={detected_lang!r}")

    diar_segs = diarize_audio(audio_path, num_speakers=num_speakers, max_speakers=max_speakers)
    labeled = _assign_speakers(whisper_segs, diar_segs)
    return _format_diarized_transcript(labeled)


def load_whisper_model(model_name: str):
    import os

    from faster_whisper import WhisperModel

    device = os.getenv("WHISPER_DEVICE", "auto")
    compute = "float16" if device == "cuda" else "int8"
    model_path = os.getenv("WHISPER_MODEL_PATH") or model_name
    print(f"Loading whisper model {model_path!r} (device={device}) ...")
    return WhisperModel(model_path, device=device, compute_type=compute)


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

    p.add_argument("--whisper-model", default="medium", help="faster-whisper model size or name")
    p.add_argument("--language", default=None, help="Force transcription language (e.g. zh); auto-detect if unset")
    p.add_argument("--chunk-chars", type=int, default=4000, help="Max characters per memory chunk (0 = no split)")
    p.add_argument("--no-infer", dest="infer", action="store_false", default=True, help="Store raw transcript text verbatim instead of LLM fact extraction")

    # --- Isolation knobs: keep podcast memories separate from broker memories ---
    p.add_argument("--user-id", default="xiaoyuzhou", help="TeleMem user_id namespace for these episodes")
    p.add_argument("--collection", default="xiaoyuzhou", help="Dedicated vector-store collection name")
    p.add_argument("--vector-path", default="db/faiss_db_podcast", help="Dedicated FAISS directory")
    p.add_argument("--history-db", default="db/history_podcast.db", help="Dedicated history DB path")

    p.add_argument("--work-dir", default="./data/podcast_audio", help="Where the processed-state file is kept")

    # --- Speaker diarization ---
    p.add_argument("--diarize", action="store_true", default=False,
                   help="Identify individual speakers (offline; requires resemblyzer + scikit-learn)")
    p.add_argument("--num-speakers", type=int, default=None,
                   help="Known number of speakers; improves accuracy when set (auto-detected otherwise)")
    p.add_argument("--max-speakers", type=int, default=8,
                   help="Upper bound for auto speaker count detection (default: 8)")

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

    whisper = load_whisper_model(args.whisper_model)
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
                if args.diarize:
                    text = transcribe_audio_diarized(
                        whisper, str(dest),
                        language=args.language,
                        num_speakers=args.num_speakers,
                        max_speakers=args.max_speakers,
                    )
                else:
                    text = transcribe_audio(whisper, str(dest), language=args.language)
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
                        "diarized": args.diarize,
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
