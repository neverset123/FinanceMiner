"""Query TeleMem memory built from a YouTube channel's transcripts (Route A).

Usage examples
--------------
    # semantic search over one channel's ingested transcripts
    python query_memory.py --user-id Bloomberg --query "what did they say about rate cuts?"

    # print an LLM-ready context block instead of a bare list
    TELEMEM_CONFIG=../../telemem/config/config.yaml \
        python query_memory.py --user-id Bloomberg --query "inflation outlook" --as-context

The ``--as-context`` mode formats the top hits into a prompt block you can paste
in front of a user question so an LLM answers grounded in the channel's content.
"""

from __future__ import annotations

import argparse
import os
import sys


def _make_memory(config_path: str | None):
    import telemem as mem0

    if config_path:
        from telemem.utils import load_config

        return mem0.Memory(config=load_config(config_path))
    return mem0.Memory()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user-id", required=True, help="TeleMem user_id used during ingestion")
    p.add_argument("--query", required=True, help="Natural-language question to search for")
    p.add_argument("--top-k", type=int, default=5, help="Number of memories to retrieve")
    p.add_argument("--config", default=os.getenv("TELEMEM_CONFIG"), help="Path to a TeleMem config YAML")
    p.add_argument("--as-context", action="store_true", help="Emit an LLM-ready context block")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    memory = _make_memory(args.config)

    results = memory.search(args.query, user_id=args.user_id, limit=args.top_k)
    hits = results.get("results", []) if isinstance(results, dict) else results

    if not hits:
        print("No matching memories found.")
        return 0

    if args.as_context:
        print("Use the following channel context to answer the question.\n")
        print("<context>")
        for i, hit in enumerate(hits, 1):
            text = hit.get("memory", "")
            meta = hit.get("metadata") or {}
            src = meta.get("title") or meta.get("video_id") or ""
            url = meta.get("url", "")
            print(f"[{i}] {text}".strip())
            if src or url:
                print(f"    (source: {src} {url})".rstrip())
        print("</context>\n")
        print(f"Question: {args.query}")
        return 0

    for i, hit in enumerate(hits, 1):
        score = hit.get("score")
        text = hit.get("memory", "")
        meta = hit.get("metadata") or {}
        head = f"{i}. " + (f"[{score:.3f}] " if isinstance(score, (int, float)) else "")
        print(head + text)
        if meta:
            print(f"     {meta.get('title', '')} {meta.get('url', '')}".rstrip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
