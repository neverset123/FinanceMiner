"""Read the latest N records from history.db and produce an LLM-summarised report.

Connects to the TeleMem history database, fetches recent memory-event records,
groups them by time window, and calls an LLM to produce a human-readable
Markdown summary.

Usage
-----
    python scripts/broker_blog.py

    # override defaults
    python scripts/broker_blog.py \
        --db db/history.db \
        --limit 20 \
        --output report.md
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent


def fetch_history(db_path: str | Path, limit: int) -> list[dict]:
    """Return the latest *limit* rows from the history table as dicts."""
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Check if upload_date column exists (added by ingest_channel.py)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(history)")}
    if "upload_date" in cols:
        cur = conn.execute(
            "SELECT id, memory_id, old_memory, new_memory, event, "
            "       created_at, updated_at, is_deleted, actor_id, role, upload_date "
            "FROM history ORDER BY upload_date DESC LIMIT ?",
            (limit,),
        )
    else:
        cur = conn.execute(
            "SELECT id, memory_id, old_memory, new_memory, event, "
            "       created_at, updated_at, is_deleted, actor_id, role "
            "FROM history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


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


SYSTEM_PROMPT = """\
You are a financial research analyst.
You receive a batch of memory records ingested from financial media channels.
Each record contains a snippet of transcript text, an event type, and a timestamp.

Your task: produce a concise, well-structured Markdown report that summarises
the key themes, market insights, and actionable takeaways from these records.

Output format (strictly follow):
## Key Themes
- Bullet list of the 3-5 most important themes across all records

## Market Insights
For each major topic, write a short paragraph (2-4 sentences) explaining the
market situation, relevant data points, and implications.

## Actionable Takeaways
- Bullet list of concrete, actionable points for an investor

## Timeline
Brief chronological summary of when the information was ingested.

Write in the same language as the source material.
Keep the total report under 800 words.
"""


def format_records_for_llm(rows: list[dict]) -> str:
    """Build a text block summarising the raw DB rows for the LLM."""
    lines = []
    for i, r in enumerate(rows, 1):
        text = r["new_memory"] or r["old_memory"] or "(empty)"
        ts = r["created_at"] or "unknown"
        upload = r.get("upload_date") or ""
        header = f"[{i}] event={r['event']} | time={ts}"
        if upload:
            header += f" | upload_date={upload}"
        lines.append(f"{header}\n{text}\n")
    return "\n".join(lines)


def render_report_header(rows: list[dict], db_path: str) -> str:
    """Return the Markdown header section (before the LLM summary)."""
    if not rows:
        return "# History Report\n\nNo records found.\n"

    earliest = rows[-1]["created_at"] or "?"
    latest = rows[0]["created_at"] or "?"
    events = {}
    for r in rows:
        events[r["event"]] = events.get(r["event"], 0) + 1

    lines = [
        "# History Report\n",
        "## Overview\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Database | `{db_path}` |",
        f"| Records shown | {len(rows)} |",
        f"| Time range | {earliest} — {latest} |",
        f"| Events | {', '.join(f'{k}: {v}' for k, v in sorted(events.items()))} |",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--db",
        default=str(_REPO_ROOT / "db" / "history.db"),
        help="Path to history.db (default: db/history.db)",
    )
    p.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Number of latest records to read (default: 10)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Save report to this file (default: print to stdout)",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Skip LLM summarisation; just dump records",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- fetch ---
    print(f"Reading latest {args.limit} records from: {args.db}", file=sys.stderr)
    rows = fetch_history(args.db, args.limit)
    print(f"  {len(rows)} records fetched.", file=sys.stderr)

    if not rows:
        print("No records found.", file=sys.stderr)
        return 0

    header = render_report_header(rows, args.db)

    if args.raw:
        # Dump without LLM
        raw_section = "## Raw Records\n\n" + format_records_for_llm(rows)
        report = header + "\n" + raw_section
    else:
        # --- LLM summary ---
        print("Building LLM client...", file=sys.stderr)
        client, model = build_openai_client()
        print(f"  LLM: {model}", file=sys.stderr)

        record_text = format_records_for_llm(rows)
        print("Generating summary...", file=sys.stderr)
        summary = call_llm(client, model, SYSTEM_PROMPT, record_text)
        report = header + "\n" + summary

    # --- output ---
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report saved to: {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
