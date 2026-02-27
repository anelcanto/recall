"""Unit tests for recall MCP tools (no services required)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest


def make_mock_client(response_data: dict) -> MagicMock:
    """Create a MagicMock httpx.Client whose context-manager methods return the given data."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = False
    mock_client.post.return_value = mock_resp
    mock_client.get.return_value = mock_resp
    mock_client.delete.return_value = mock_resp
    return mock_client


class TestStoreMemory:
    def test_store_memory(self):
        expected = {"id": "abc-123", "id_strategy": "uuid"}
        mock_client = make_mock_client(expected)
        with patch("cli.recall_mcp._client", return_value=mock_client):
            from cli.recall_mcp import store_memory

            result = store_memory("Remember this")
        assert result == expected
        mock_client.post.assert_called_once_with(
            "/memory",
            json={"text": "Remember this", "tags": [], "source": "claude", "dedupe_key": None},
        )

    def test_store_memory_with_dedupe(self):
        expected = {"id": "abc-456", "id_strategy": "dedupe_key"}
        mock_client = make_mock_client(expected)
        with patch("cli.recall_mcp._client", return_value=mock_client):
            from cli.recall_mcp import store_memory

            result = store_memory("Hello", dedupe_key="my-key")
        assert result == expected
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["dedupe_key"] == "my-key"

    def test_http_error_propagates(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_error_response.text = "Internal Server Error"
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=mock_error_response,
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_resp

        with patch("cli.recall_mcp._client", return_value=mock_client):
            from cli.recall_mcp import store_memory

            with pytest.raises(httpx.HTTPStatusError):
                store_memory("bad request")


class TestSearchMemories:
    def test_search_memories(self):
        expected = {"results": [{"id": "id1", "score": 0.9, "text": "hello"}]}
        mock_client = make_mock_client(expected)
        with patch("cli.recall_mcp._client", return_value=mock_client):
            from cli.recall_mcp import search_memories

            result = search_memories("hello", top_k=3)
        assert result == expected
        mock_client.post.assert_called_once_with(
            "/search",
            json={"query": "hello", "top_k": 3, "include_text": True},
        )


class TestListMemories:
    def test_list_memories(self):
        expected = {"memories": [], "next_cursor": None}
        mock_client = make_mock_client(expected)
        with patch("cli.recall_mcp._client", return_value=mock_client):
            from cli.recall_mcp import list_memories

            result = list_memories(limit=10)
        assert result == expected
        mock_client.get.assert_called_once_with("/memories", params={"limit": 10})


class TestDeleteMemory:
    def test_delete_memory(self):
        expected = {"message": "deleted"}
        mock_client = make_mock_client(expected)
        with patch("cli.recall_mcp._client", return_value=mock_client):
            from cli.recall_mcp import delete_memory

            result = delete_memory("mem-id-123")
        assert result == expected
        mock_client.delete.assert_called_once_with("/memory/mem-id-123")


class TestCheckHealth:
    def test_check_health(self):
        expected = {"status": "ok", "qdrant": "healthy", "ollama": "healthy"}
        mock_client = make_mock_client(expected)
        with patch("cli.recall_mcp._client", return_value=mock_client):
            from cli.recall_mcp import check_health

            result = check_health()
        assert result == expected
        mock_client.get.assert_called_once_with("/health")
