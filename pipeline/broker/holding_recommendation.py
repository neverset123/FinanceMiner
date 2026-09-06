"""Holding recommendation engine.

Loads current holdings from a broker JSON snapshot, queries the TeleMem vector
store for relevant market intelligence per position, then calls an LLM to
produce a structured BUY / HOLD / REDUCE / SELL recommendation for each stock.

Usage
-----
    python pipeline/broker/holding_recommendation.py

    # override defaults
    python pipeline/broker/holding_recommendation.py \\
        --holdings data/sc/json/holdings.json \\
        --user-id bellafinance \\
        --top-k 5 \\
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

def load_holdings(path: str | Path) -> list[dict]:
    """Parse broker holdings JSON and return a list of position dicts."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    # Unwrap {ok, command, data: {result: {items: [...]}}}
    try:
        items = raw["data"]["result"]["items"]
    except (KeyError, TypeError):
        # Accept a bare list as well
        items = raw if isinstance(raw, list) else []
    return items


def enrich_position(pos: dict) -> dict:
    """Add derived P&L fields to a position dict."""
    fifo = pos.get("fifo_price", 0.0)
    current = pos.get("quote_mid_price", 0.0)
    qty = pos.get("quantity", 0)

    unrealised_pct = ((current - fifo) / fifo * 100) if fifo else 0.0
    unrealised_eur = (current - fifo) * qty

    return {
        **pos,
        "unrealised_pct": unrealised_pct,
        "unrealised_eur": unrealised_eur,
        "cost_basis_eur": fifo * qty,
    }


def position_summary_line(pos: dict) -> str:
    sign = "+" if pos["unrealised_eur"] >= 0 else ""
    return (
        f"{pos['name']} ({pos['isin']}): "
        f"qty={pos['quantity']} | "
        f"cost={pos['fifo_price']:.2f} | "
        f"current={pos['quote_mid_price']:.2f} | "
        f"P&L={sign}{pos['unrealised_eur']:.0f} EUR "
        f"({sign}{pos['unrealised_pct']:.1f}%) | "
        f"valuation={pos['valuation']:.0f} EUR"
    )


def query_telemem(memory, query: str, user_id: str, top_k: int) -> str:
    """Return a formatted context string from TeleMem for *query*."""
    try:
        results = memory.search(query, user_id=user_id, limit=top_k)
        hits = results.get("results", []) if isinstance(results, dict) else results
    except Exception as exc:
        return f"(TeleMem query failed: {exc})"

    if not hits:
        return "(no relevant memories found)"

    lines = []
    for hit in hits:
        text = hit.get("memory", "").strip()
        meta = hit.get("metadata") or {}
        src = meta.get("title") or meta.get("video_id") or ""
        if text:
            lines.append(f"- {text}" + (f"  [{src}]" if src else ""))
    return "\n".join(lines) if lines else "(no relevant memories found)"


def build_openai_client(llm_cfg: dict):
    """Construct an openai.OpenAI client from a telemem-style llm config dict.

    Mirrors agent/llm.py: uses httpx with SSL-bypass when behind a self-signed
    proxy (NODE_TLS_REJECT_UNAUTHORIZED=0).
    """
    import httpx
    from openai import OpenAI

    cfg = llm_cfg.get("config", {})
    verify = os.getenv("NODE_TLS_REJECT_UNAUTHORIZED", "1") != "0"
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or None
    http_client = httpx.Client(verify=verify, proxy=proxy)

    return OpenAI(
        api_key=cfg.get("api_key", "sk-xxx"),
        base_url=cfg.get("openai_base_url", "http://localhost:8081/v1"),
        http_client=http_client,
    ), cfg.get("model", "qwen3-8b")


def call_llm(client, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


SYSTEM_PROMPT = """You are an expert equity analyst.
Given a single stock position (P&L, cost basis, current price, valuation) and
relevant market/channel context from a financial media channel, produce a
concise investment recommendation.

Output format (strictly follow):
RECOMMENDATION: <BUY | ADD | HOLD | REDUCE | SELL>
RATIONALE: <2-4 sentences summarising the key drivers>
RISK: <1-2 sentences on main downside risk>
TARGET_ACTION: <specific, actionable suggestion, e.g. "hold, set stop-loss at X" or "sell 50% to lock in gains">
"""


def recommend_position(client, model: str, pos: dict, context: str) -> dict:
    user_msg = f"""### Position
{position_summary_line(pos)}

### Market & Channel Context
{context}

Provide your recommendation."""

    raw = call_llm(client, model, SYSTEM_PROMPT, user_msg)

    # Parse structured fields
    rec = {"name": pos["name"], "isin": pos["isin"], "raw": raw}
    for field in ("RECOMMENDATION", "RATIONALE", "RISK", "TARGET_ACTION"):
        prefix = f"{field}:"
        for line in raw.splitlines():
            if line.strip().upper().startswith(prefix):
                rec[field.lower()] = line.split(":", 1)[1].strip()
                break
        else:
            rec[field.lower()] = ""
    return rec


PORTFOLIO_SYSTEM = """You are a senior portfolio manager.
Given a list of individual stock recommendations and overall portfolio metrics,
write a brief portfolio-level strategy summary (max 200 words).
Focus on: concentration risk, sector balance, largest P&L impacts, and
top 3 priority actions for maximum economic profit.
"""


def portfolio_summary(client, model: str, positions: list[dict], recs: list[dict]) -> str:
    total_value = sum(p["valuation"] for p in positions)
    total_cost = sum(p["cost_basis_eur"] for p in positions)
    total_pnl = sum(p["unrealised_eur"] for p in positions)
    pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0.0

    rec_lines = "\n".join(
        f"- {r['name']}: {r.get('recommendation','?').upper()} — {r.get('target_action','')}"
        for r in recs
    )

    user_msg = f"""### Portfolio Snapshot
Total value: {total_value:,.0f} EUR
Total cost: {total_cost:,.0f} EUR
Unrealised P&L: {'+' if total_pnl>=0 else ''}{total_pnl:,.0f} EUR ({'+' if pnl_pct>=0 else ''}{pnl_pct:.1f}%)
Positions: {len(positions)}

### Individual Recommendations
{rec_lines}

Write the portfolio strategy summary."""

    return call_llm(client, model, PORTFOLIO_SYSTEM, user_msg)


RATING_EMOJI = {
    "BUY": "🟢", "ADD": "🟢",
    "HOLD": "🟡",
    "REDUCE": "🟠", "SELL": "🔴",
}


def render_report(positions: list[dict], recs: list[dict], summary: str) -> str:
    lines = []
    lines.append("# Holding Recommendation Report\n")

    # Portfolio overview table
    total_value = sum(p["valuation"] for p in positions)
    total_cost = sum(p["cost_basis_eur"] for p in positions)
    total_pnl = sum(p["unrealised_eur"] for p in positions)
    pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0.0
    lines.append("## Portfolio Overview\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total Value | {total_value:,.0f} EUR |")
    lines.append(f"| Total Cost  | {total_cost:,.0f} EUR |")
    sign = "+" if total_pnl >= 0 else ""
    lines.append(f"| Unrealised P&L | {sign}{total_pnl:,.0f} EUR ({sign}{pnl_pct:.1f}%) |")
    lines.append(f"| Positions | {len(positions)} |\n")

    # Individual recommendations sorted by |P&L EUR| descending
    lines.append("## Position Recommendations\n")
    sorted_recs = sorted(recs, key=lambda r: abs(next(
        (p["unrealised_eur"] for p in positions if p["isin"] == r["isin"]), 0
    )), reverse=True)

    for r in sorted_recs:
        pos = next((p for p in positions if p["isin"] == r["isin"]), {})
        rating = r.get("recommendation", "?").upper()
        emoji = RATING_EMOJI.get(rating, "⚪")
        sign = "+" if pos.get("unrealised_eur", 0) >= 0 else ""
        lines.append(f"### {emoji} {r['name']} — {rating}\n")
        lines.append(f"- **P&L**: {sign}{pos.get('unrealised_eur',0):,.0f} EUR "
                     f"({sign}{pos.get('unrealised_pct',0):.1f}%)  "
                     f"| Valuation: {pos.get('valuation',0):,.0f} EUR")
        if r.get("rationale"):
            lines.append(f"- **Rationale**: {r['rationale']}")
        if r.get("risk"):
            lines.append(f"- **Risk**: {r['risk']}")
        if r.get("target_action"):
            lines.append(f"- **Action**: {r['target_action']}")
        lines.append("")

    # Portfolio summary
    lines.append("## Portfolio Strategy\n")
    lines.append(summary)

    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--holdings",
        default=str(_REPO_ROOT / "data" / "sc" / "json" / "holdings.json"),
        help="Path to broker holdings JSON (default: data/sc/json/holdings.json)",
    )
    p.add_argument(
        "--user-id",
        default="bellafinance",
        help="TeleMem user_id to query (default: bellafinance)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="TeleMem hits per stock (default: 3)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Save report to this file (default: print to stdout)",
    )
    p.add_argument(
        "--config",
        default=os.getenv("TELEMEM_CONFIG"),
        help="Path to a TeleMem config YAML (optional)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # --- holdings ---
    print(f"Loading holdings from: {args.holdings}", file=sys.stderr)
    raw_positions = load_holdings(args.holdings)
    positions = [enrich_position(p) for p in raw_positions]
    print(f"  {len(positions)} positions loaded.", file=sys.stderr)

    # --- telemem ---
    print("Initialising TeleMem memory...", file=sys.stderr)
    sys.path.insert(0, str(_HERE))
    from config import make_memory, build_telemem_config, _resolve_llm  # noqa: E402

    memory = make_memory(args.config)
    print("  TeleMem ready.", file=sys.stderr)

    # --- LLM client ---
    llm_cfg = _resolve_llm()
    client, model = build_openai_client(llm_cfg)
    print(f"  LLM: {model}", file=sys.stderr)

    # --- per-position recommendations ---
    recs = []
    for i, pos in enumerate(positions, 1):
        name = pos["name"]
        print(f"[{i}/{len(positions)}] Analysing {name}...", file=sys.stderr)

        # Query telemem with company name + ISIN + ticker keywords
        query = f"{name} stock outlook earnings valuation"
        context = query_telemem(memory, query, args.user_id, args.top_k)

        rec = recommend_position(client, model, pos, context)
        recs.append(rec)
        print(f"  → {rec.get('recommendation', '?').upper()}", file=sys.stderr)

    # --- portfolio summary ---
    print("Generating portfolio summary...", file=sys.stderr)
    summary = portfolio_summary(client, model, positions, recs)

    # --- render ---
    report = render_report(positions, recs, summary)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report saved to: {args.output}", file=sys.stderr)
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
