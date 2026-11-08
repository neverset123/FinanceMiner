"""Delete the most recently ingested podcast episode from the podcast TeleMem store.

Removes *all* memory chunks belonging to one episode (not just a single row),
and optionally un-marks it in the ingest state file so ``ingest_podcast.py``
will process it again on the next run.

The podcast memories live in their own store (collection ``xiaoyuzhou``,
``db/faiss_db_podcast``, ``db/history_podcast.db``), so this script builds an
isolated memory exactly like ``ingest_podcast.py`` does. Note the stored
``user_id`` is ``crossing`` (the collection name is a separate thing).

Usage
-----
    cd pipeline/broker

    # show what would be deleted
    python delete_podcast_episode.py --dry-run

    # delete the latest episode's chunks and allow re-ingestion
    python delete_podcast_episode.py

    # delete a specific episode instead of the latest
    python delete_podcast_episode.py --episode-id 6ae6e230c96c8942ec156ed6

    # keep the state file untouched (episode stays "processed")
    python delete_podcast_episode.py --keep-state
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ingest_podcast import build_isolated_memory


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--user-id", default="crossing", help="TeleMem user_id namespace (default: crossing)")
    p.add_argument("--collection", default="xiaoyuzhou", help="Vector-store collection name")
    p.add_argument("--vector-path", default="db/faiss_db_podcast", help="FAISS directory")
    p.add_argument("--history-db", default="db/history_podcast.db", help="History DB path")
    p.add_argument("--work-dir", default="./data/podcast_audio", help="Where state_<user-id>.json lives")
    p.add_argument(
        "--episode-id",
        default=None,
        help="Episode to delete; defaults to the most recently ingested one",
    )
    p.add_argument("--dry-run", action="store_true", help="Show what would be deleted, change nothing")
    p.add_argument(
        "--keep-state",
        action="store_true",
        help="Do not remove the episode from the processed-state file",
    )
    return p.parse_args(argv)


def _rows(result) -> list[dict]:
    if isinstance(result, dict):
        return result.get("results", []) or []
    return list(result or [])


def _get_all(memory, user_id: str, limit: int = 10000):
    """Call ``get_all`` across mem0 versions.

    Newer mem0 rejects top-level entity params and requires ``filters={...}``,
    and caps results with ``top_k`` (default 20); older versions used ``limit``
    and a top-level ``user_id``. Try each form so no chunks are silently hidden.
    """
    attempts = (
        lambda: memory.get_all(filters={"user_id": user_id}, top_k=limit),
        lambda: memory.get_all(filters={"user_id": user_id}, limit=limit),
        lambda: memory.get_all(filters={"user_id": user_id}),
        lambda: memory.get_all(user_id=user_id, limit=limit),
    )
    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except (TypeError, ValueError) as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def _meta(row: dict) -> dict:
    """Return row metadata, tolerating stores that keep fields flat on the row."""
    meta = row.get("metadata")
    if isinstance(meta, dict) and meta:
        merged = {k: v for k, v in row.items() if k not in ("metadata", "memory", "id")}
        merged.update(meta)
        return merged
    return {k: v for k, v in row.items() if k not in ("metadata", "memory", "id")}


def _sort_key(row: dict):
    meta = _meta(row)
    return (
        row.get("created_at") or meta.get("created_at") or row.get("updated_at") or "",
        meta.get("chunk") if isinstance(meta.get("chunk"), int) else 0,
    )


def pick_latest_episode(rows: list[dict]) -> str | None:
    """Return the episode_id of the newest row that carries one."""
    for row in sorted(rows, key=_sort_key, reverse=True):
        ep = _meta(row).get("episode_id")
        if ep:
            return ep
    return None


def clear_state_entry(state_path: Path, episode_id: str) -> bool:
    import yt_utils

    state = yt_utils.load_state(str(state_path))
    processed = list(state.get("processed", []))
    if episode_id not in processed:
        return False
    processed.remove(episode_id)
    state["processed"] = sorted(processed)
    yt_utils.save_state(str(state_path), state)
    return True


def main(argv=None) -> int:
    args = parse_args(argv)

    memory = build_isolated_memory(args.collection, args.vector_path, args.history_db)
    print(
        f"Store: collection={args.collection!r} vector_path={args.vector_path!r} "
        f"history_db={args.history_db!r} user_id={args.user_id!r}"
    )

    print("Loading memories (this can take a moment on large FAISS stores) ...", flush=True)
    rows = _rows(_get_all(memory, args.user_id))
    print(f"Loaded {len(rows)} memory row(s).", flush=True)
    if not rows:
        print("No memories found for this user_id; nothing to do.")
        return 0

    episode_id = args.episode_id or pick_latest_episode(rows)
    if not episode_id:
        print("Could not determine an episode_id from memory metadata.", file=sys.stderr)
        return 1

    targets = [r for r in rows if _meta(r).get("episode_id") == episode_id]
    if not targets:
        print(f"No memories tagged episode_id={episode_id!r}.", file=sys.stderr)
        return 1

    title = _meta(targets[0]).get("title") or ""
    print(f"\nEpisode : {episode_id}")
    if title:
        print(f"Title   : {title}")
    print(f"Chunks  : {len(targets)} of {len(rows)} total memories\n")

    for row in sorted(targets, key=_sort_key):
        meta = _meta(row)
        preview = (row.get("memory") or meta.get("data") or "").replace("\n", " ")[:90]
        print(f"  - {row.get('id')} [chunk {meta.get('chunk')}] {preview}")

    if args.dry_run:
        print("\n--dry-run: nothing deleted.")
        return 0

    deleted = 0
    for row in targets:
        mem_id = row.get("id")
        try:
            try:
                memory.delete(memory_id=mem_id)
            except TypeError:
                memory.delete(mem_id)
            deleted += 1
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            print(f"  ! failed to delete {mem_id}: {exc}", file=sys.stderr)

    print(f"\nDeleted {deleted}/{len(targets)} memory chunk(s).")

    if not args.keep_state:
        state_path = Path(args.work_dir).expanduser().resolve() / f"state_{args.user_id}.json"
        if clear_state_entry(state_path, episode_id):
            print(f"Removed {episode_id} from {state_path} (will be re-ingested).")
        else:
            print(f"{episode_id} was not listed in {state_path}.")

    return 0 if deleted == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
