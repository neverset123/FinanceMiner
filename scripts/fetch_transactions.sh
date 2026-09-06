#!/usr/bin/env bash
# Fetch all broker transactions from Scalable Capital CLI and save to a single JSON file.
# Usage: ./scripts/fetch_transactions.sh [--from-time 2024-01-01T00:00:00Z] [--to-time ...]
#
# The script paginates automatically (100 per page) and merges all results into
# data/sc/json/transactions.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/data/sc/json"
OUT_FILE="$OUT_DIR/transactions.json"
PAGE_SIZE=100
EXTRA_ARGS=("$@")

mkdir -p "$OUT_DIR"

all_items="[]"
cursor=""
page=1

while true; do
    echo "Fetching page $page (cursor=${cursor:-<none>})..."

    cmd=(sc broker transactions --page-size "$PAGE_SIZE" --type-filter DEPOSIT --type-filter WITHDRAWAL --json)
    if [[ -n "$cursor" ]]; then
        cmd+=(--cursor "$cursor")
    fi
    # Append any extra args the caller passed (e.g. --from-time, --to-time)
    cmd+=("${EXTRA_ARGS[@]}")

    response=$("${cmd[@]}" 2>&1)

    ok=$(echo "$response" | jq -r '.ok')
    if [[ "$ok" != "true" ]]; then
        echo "Error from sc CLI:"
        echo "$response" | jq .
        exit 1
    fi

    items=$(echo "$response" | jq '.data.result.items')
    count=$(echo "$items" | jq 'length')
    all_items=$(jq -n --argjson a "$all_items" --argjson b "$items" '$a + $b')

    echo "  Got $count transactions (total so far: $(echo "$all_items" | jq 'length'))"

    cursor=$(echo "$response" | jq -r '.data.result.cursor // empty')
    if [[ -z "$cursor" ]]; then
        echo "No more pages."
        break
    fi

    page=$((page + 1))
done

total=$(echo "$all_items" | jq 'length')

jq -n --argjson items "$all_items" '{total: ($items | length), transactions: $items}' > "$OUT_FILE"

echo "Done. Saved $total transactions to $OUT_FILE"
