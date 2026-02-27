"""Unit tests for OllamaClient embeddings (no services required)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from memory_api.embeddings import EmbeddingUnavailable, OllamaClient


def make_async_response(status_code: int, body: dict) -> MagicMock:
    """Build a fake async httpx response mock."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    if status_code >= 400:
        mock_error_response = MagicMock()
        mock_error_response.status_code = status_code
        mock_error_response.text = "error"
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_error_response,
        )
    else:
        resp.raise_for_status = MagicMock()
    return resp


def make_async_client(response: MagicMock) -> MagicMock:
    """Wrap a response in an async httpx.AsyncClient mock."""
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    return client


class TestGetEmbedding:
    async def test_get_embedding_new_api(self):
        """POST /api/embed returns {"embeddings": [[...]]}."""
        vec = [0.1, 0.2, 0.3]
        resp = make_async_response(200, {"embeddings": [vec]})
        client = OllamaClient("http://localhost:11434", "nomic-embed-text", "/api/embed")
        client.set_client(make_async_client(resp))
        result = await client.get_embedding("hello")
        assert result == vec

    async def test_get_embedding_old_api(self):
        """POST /api/embeddings returns {"embedding": [...]}."""
        vec = [0.4, 0.5, 0.6]
        resp = make_async_response(200, {"embedding": vec})
        client = OllamaClient("http://localhost:11434", "nomic-embed-text", "/api/embeddings")
        client.set_client(make_async_client(resp))
        result = await client.get_embedding("hello")
        assert result == vec

    async def test_no_client_raises(self):
        """Calling get_embedding without set_client raises EmbeddingUnavailable."""
        client = OllamaClient("http://localhost:11434", "nomic-embed-text", "/api/embed")
        with pytest.raises(EmbeddingUnavailable, match="not initialized"):
            await client.get_embedding("hello")

    async def test_path_autodetect_fallback(self):
        """If the primary path fails with 404, falls back to the secondary path."""
        vec = [0.7, 0.8]
        resp_fail = make_async_response(404, {})
        resp_ok = make_async_response(200, {"embeddings": [vec]})

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=[resp_fail, resp_ok])

        client = OllamaClient("http://localhost:11434", "nomic-embed-text", "/api/embed")
        client.set_client(mock_client)
        result = await client.get_embedding("test")
        assert result == vec
        assert client._working_path == "/api/embeddings"

    async def test_path_cached_after_success(self):
        """Second call uses the cached working path — no re-detection."""
        vec = [0.1, 0.2]
        resp = make_async_response(200, {"embeddings": [vec]})
        mock_post = AsyncMock(return_value=resp)
        mock_client = MagicMock()
        mock_client.post = mock_post

        client = OllamaClient("http://localhost:11434", "nomic-embed-text", "/api/embed")
        client.set_client(mock_client)
        await client.get_embedding("first")
        await client.get_embedding("second")
        assert mock_post.call_count == 2


class TestProbeDimension:
    async def test_probe_dimension(self):
        vec = [0.1] * 768
        resp = make_async_response(200, {"embeddings": [vec]})
        client = OllamaClient("http://localhost:11434", "nomic-embed-text", "/api/embed")
        client.set_client(make_async_client(resp))
        dim = await client.probe_dimension()
        assert dim == 768

    async def test_probe_dimension_cached(self):
        """Second probe_dimension call returns cached value without making an HTTP call."""
        vec = [0.1] * 384
        mock_post = AsyncMock(return_value=make_async_response(200, {"embeddings": [vec]}))
        mock_client = MagicMock()
        mock_client.post = mock_post

        client = OllamaClient("http://localhost:11434", "nomic-embed-text", "/api/embed")
        client.set_client(mock_client)

        first = await client.probe_dimension()
        second = await client.probe_dimension()
        assert first == second == 384
        assert mock_post.call_count == 1


class TestIsAvailable:
    async def test_is_available_true(self):
        vec = [0.1, 0.2]
        resp = make_async_response(200, {"embeddings": [vec]})
        client = OllamaClient("http://localhost:11434", "nomic-embed-text", "/api/embed")
        client.set_client(make_async_client(resp))
        assert await client.is_available() is True

    async def test_is_available_false(self):
        """No HTTP client set → is_available returns False."""
        client = OllamaClient("http://localhost:11434", "nomic-embed-text", "/api/embed")
        assert await client.is_available() is False
