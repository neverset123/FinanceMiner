"""Read podcast episodes from the FAISS docstore and produce an LLM-summarised report.

Reads the FAISS docstore, groups chunks by episode, and calls an LLM to
produce a per-episode Markdown summary.

Usage
-----
    python pipeline/broker/broker_blog.py \
        --faiss-json db/faiss_db_podcast/xiaoyuzhou.json \
        --user-id crossing \
        --limit 3 \
        --output report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent


def fetch_episodes_from_docstore(
    faiss_json: Path,
    user_id: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Read the FAISS docstore and return complete episodes grouped by episode_id.

    Each episode dict has:
        episode_id, title, url, created_at,
        chunks: [{"chunk": int, "text": str}, ...]  sorted by chunk index.

    Episodes are ordered newest-first (by the first chunk's created_at).
    """
    if not faiss_json.exists():
        print(f"ERROR: FAISS docstore not found: {faiss_json}", file=sys.stderr)
        sys.exit(1)

    with open(faiss_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    docstore: dict = data.get("docstore", {}) if isinstance(data, dict) else {}

    episodes: dict[str, dict] = {}
    for meta in docstore.values():
        if not isinstance(meta, dict):
            continue
        ep_id = meta.get("episode_id")
        if not ep_id:
            continue
        if user_id and meta.get("user_id") != user_id:
            continue

        if ep_id not in episodes:
            episodes[ep_id] = {
                "episode_id": ep_id,
                "title": meta.get("title", ep_id),
                "url": meta.get("url", ""),
                "created_at": meta.get("created_at", ""),
                "chunks": [],
            }
        episodes[ep_id]["chunks"].append({
            "chunk": meta.get("chunk", 0),
            "text": meta.get("data", ""),
        })

    for ep in episodes.values():
        ep["chunks"].sort(key=lambda c: c["chunk"])

    sorted_eps = sorted(
        episodes.values(),
        key=lambda e: e["created_at"],
        reverse=True,
    )
    return sorted_eps[:limit] if limit else sorted_eps


def build_openai_client():
    """Construct an OpenAI client using the broker config helpers."""
    import httpx
    from openai import OpenAI
    from config import _resolve_llm  # noqa: E402

    llm_cfg = _resolve_llm()
    cfg = llm_cfg.get("config", {})

    verify = os.getenv("NODE_TLS_REJECT_UNAUTHORIZED", "1") != "0"
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None
    http_client = httpx.Client(verify=verify, proxy=proxy)

    client = OpenAI(
        api_key=cfg.get("api_key", "sk-xxx"),
        base_url=cfg.get("openai_base_url", "http://localhost:8081/v1"),
        http_client=http_client,
    )
    model = cfg.get("model", "qwen3-8b")
    return client, model


def call_llm(client, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# --- Summarization prompts (map-reduce) ---

_EXTRACT_PROMPT = """\
You are reading a segment of a podcast transcript.
Extract EVERY topic discussed in this segment and the key opinions or arguments made under each topic.
Be exhaustive — do not skip any topic, even if it seems minor.

Output format (repeat for each topic found):
### <Topic Name>
- <Opinion, argument, or key point>
- <Another point>

Write in the same language as the transcript.
Output only the topic/opinion structure with no preamble or closing remarks.
"""

_MERGE_PROMPT = """\
You are given topic-and-opinion extractions from consecutive segments of the same podcast episode.
Merge them into one coherent, complete summary:
- Combine duplicate or closely related topics under a single heading
- Preserve EVERY unique opinion, argument, and detail — nothing should be dropped
- Use nested bullet points for sub-discussions within a topic
- Keep headings concise but descriptive

Output format (repeat for each topic):
### <Topic Name>
- <Key opinion or point>
  - <Supporting detail if needed>
- <Another point>

Write in the same language as the source.
Output only the merged topic structure with no preamble or closing remarks.
"""


def summarize_episode(client, model: str, episode: dict) -> str:
    """Produce a comprehensive topic-by-opinion summary via map-reduce.

    Each chunk is processed independently (map) to extract topics and opinions,
    then all extractions are merged into one coherent summary (reduce).
    This prevents detail loss that can occur when passing very long transcripts
    in a single LLM call.
    """
    chunks = [c["text"].strip() for c in episode["chunks"]]

    # Map: extract topics + opinions from each chunk
    print(f"  Extracting from {len(chunks)} chunk(s)...", file=sys.stderr)
    extractions: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"    chunk {i}/{len(chunks)}", file=sys.stderr)
        extraction = call_llm(client, model, _EXTRACT_PROMPT, chunk)
        if extraction.strip():
            extractions.append(extraction)

    if not extractions:
        return "_No content extracted._"

    if len(extractions) == 1:
        return extractions[0]

    # Reduce: merge all extractions into one summary
    print("  Merging extractions...", file=sys.stderr)
    combined = "\n\n---\n\n".join(
        f"[Segment {i}]\n{e}" for i, e in enumerate(extractions, 1)
    )
    return call_llm(client, model, _MERGE_PROMPT, combined)


def _join_chunks(chunks: list[str], max_overlap: int = 300) -> str:
    """Concatenate chunks, stripping the overlapping prefix from each successive one."""
    if not chunks:
        return ""
    result = chunks[0]
    for chunk in chunks[1:]:
        check = min(max_overlap, len(result), len(chunk))
        overlap = 0
        for n in range(check, 0, -1):
            if result.endswith(chunk[:n]):
                overlap = n
                break
        result += chunk[overlap:]
    return result


def format_episodes_for_llm(episodes: list[dict]) -> str:
    """Format complete episodes into a text block for the LLM."""
    lines = []
    for ep in episodes:
        lines.append(f"## {ep['title']}")
        if ep["url"]:
            lines.append(f"Source: {ep['url']}")
        lines.append("")
        text = _join_chunks([c["text"].strip() for c in ep["chunks"]])
        lines.append(text)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def render_report_header(episodes: list[dict], faiss_json: str) -> str:
    """Return the Markdown header section."""
    if not episodes:
        return "# Podcast Episode Report\n\nNo episodes found.\n"
    lines = [
        "# Podcast Episode Report\n",
        "## Overview\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Docstore | `{faiss_json}` |",
        f"| Episodes | {len(episodes)} |",
        f"| Total chunks | {sum(len(e['chunks']) for e in episodes)} |",
        "",
    ]
    for ep in episodes:
        lines.append(f"- **{ep['title']}** ({len(ep['chunks'])} chunks) — {ep['url']}")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--faiss-json",
        default=str(_REPO_ROOT / "db" / "faiss_db" / "telemem.json"),
        help="Path to the podcast FAISS docstore JSON "
             "(default: db/faiss_db/telemem.json)",
    )
    p.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Number of most recent episodes to read (default: 10)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Save report to this file (default: print to stdout)",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Skip LLM; dump full episode transcripts",
    )
    p.add_argument(
        "--user-id",
        default=None,
        help="Only include episodes ingested under this TeleMem user_id",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    faiss_json = Path(args.faiss_json)
    scope = f" (user_id={args.user_id})" if args.user_id else ""
    print(
        f"Reading up to {args.limit} episode(s) from: {faiss_json}{scope}",
        file=sys.stderr,
    )
    episodes = fetch_episodes_from_docstore(
        faiss_json,
        user_id=args.user_id,
        limit=args.limit,
    )
    print(f"  {len(episodes)} episode(s) found.", file=sys.stderr)

    if not episodes:
        print("No episodes found.", file=sys.stderr)
        return 0

    header = render_report_header(episodes, args.faiss_json)

    if args.raw:
        report = header + "\n## Raw Episodes\n\n" + format_episodes_for_llm(episodes)
    else:
        print("Building LLM client...", file=sys.stderr)
        client, model = build_openai_client()
        print(f"  LLM: {model}", file=sys.stderr)
        sections: list[str] = []
        for ep in episodes:
            print(f"Summarising: {ep['title']}", file=sys.stderr)
            ep_summary = summarize_episode(client, model, ep)
            sections.append(f"## {ep['title']}\n\n{ep_summary}")
        report = header + "\n" + "\n\n---\n\n".join(sections)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report saved to: {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
