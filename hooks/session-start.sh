#!/usr/bin/env bash
set -euo pipefail

# Load recall config (same resolution as recall_mcp.py)
ENV_FILE="$HOME/.recall/.env"
API_URL="${RECALL_API_URL:-http://127.0.0.1:8100}"
if [ -f "$ENV_FILE" ]; then
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  API_URL="${RECALL_API_URL:-http://127.0.0.1:8100}"
fi

# Health check — exit silently if recall isn't running
if ! curl -sf --max-time 3 "$API_URL/health" > /dev/null 2>&1; then
  exit 0
fi

# Fetch recent memories
RESPONSE=$(curl -sf --max-time 5 "$API_URL/memories?limit=10" 2>/dev/null) || exit 0

# Extract memory texts with jq
MEMORIES=$(echo "$RESPONSE" | jq -r '.memories[]? | "- [\(.tags | join(", "))] \(.text)"' 2>/dev/null) || exit 0

if [ -z "$MEMORIES" ]; then
  exit 0
fi

# Output as system message for Claude's context
cat <<EOF
{"systemMessage": "Recall memory context loaded. Here are the user's recent stored memories:\n\n$MEMORIES\n\nUse these for context. Do not announce them unless the user asks."}
EOF
