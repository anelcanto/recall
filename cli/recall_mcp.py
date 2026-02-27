#!/usr/bin/env python3
"""
recall MCP server — exposes recall API tools to Claude via FastMCP (stdio transport).

Config resolution (in priority order):
  1. RECALL_API_URL / RECALL_API_TOKEN environment variables
  2. ~/.recall/.env file (loaded via python-dotenv)
  3. Defaults: http://127.0.0.1:8100, no token
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_ENV_FILE = Path.home() / ".recall" / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE, override=False)  # env vars already set take priority

_API_URL = os.environ.get("RECALL_API_URL", "http://127.0.0.1:8100").rstrip("/")
_API_TOKEN = os.environ.get("RECALL_API_TOKEN") or None


def _headers() -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if _API_TOKEN:
        h["Authorization"] = f"Bearer {_API_TOKEN}"
    return h


def _client() -> httpx.Client:
    return httpx.Client(base_url=_API_URL, headers=_headers(), timeout=30)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "recall",
    instructions=(
        "Use these tools to persist and retrieve memories across Claude sessions. "
        "Search before answering questions about the user's projects or preferences. "
        "Store key decisions, user preferences, and project context proactively."
    ),
)


@mcp.tool()
def store_memory(
    text: str,
    tags: Optional[list[str]] = None,
    source: Optional[str] = "claude",
    dedupe_key: Optional[str] = None,
) -> dict:
    """Store a new memory in the recall database.

    Args:
        text: The content to remember.
        tags: Optional list of tags for organisation (e.g. ["project:recall", "preference"]).
        source: Source identifier (default: "claude").
        dedupe_key: Optional key to prevent duplicates — storing with the same key updates in place.

    Returns:
        dict with 'id' and 'id_strategy' fields.
    """
    payload = {
        "text": text,
        "tags": tags or [],
        "source": source or "claude",
        "dedupe_key": dedupe_key,
    }
    with _client() as client:
        resp = client.post("/memory", json=payload)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
def search_memories(query: str, top_k: int = 5) -> dict:
    """Search memories by semantic similarity.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of results to return (default: 5).

    Returns:
        dict with 'results' list. Each result has 'id', 'score', 'text', 'tags',
        'source', and 'written_at'.
    """
    payload = {"query": query, "top_k": top_k, "include_text": True}
    with _client() as client:
        resp = client.post("/search", json=payload)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
def list_memories(limit: int = 20) -> dict:
    """List recently stored memories.

    Args:
        limit: Maximum number of memories to return (default: 20).

    Returns:
        dict with 'memories' list and optional 'next_cursor' for pagination.
    """
    with _client() as client:
        resp = client.get("/memories", params={"limit": limit})
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
def delete_memory(memory_id: str) -> dict:
    """Delete a memory by its ID.

    Args:
        memory_id: The UUID of the memory to delete.

    Returns:
        dict with confirmation message.
    """
    with _client() as client:
        resp = client.delete(f"/memory/{memory_id}")
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
def check_health() -> dict:
    """Check the health of the recall API, Qdrant, and Ollama.

    Returns:
        dict with 'status' ("ok" | "degraded" | "error"), 'qdrant', and 'ollama' fields.
    """
    with _client() as client:
        resp = client.get("/health")
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import sys

    if "--http" in sys.argv:
        port = int(os.environ.get("RECALL_MCP_PORT", "8001"))
        mcp.run(transport="http", port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
