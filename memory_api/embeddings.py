from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class EmbeddingUnavailable(Exception):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, embed_path: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._default_embed_path = embed_path
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._working_path: Optional[str] = None
        self._dimension: Optional[int] = None

    def set_client(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def _try_embed_path(self, path: str, text: str) -> list[float]:
        assert self._client is not None
        url = self._base_url + path
        try:
            resp = await self._client.post(
                url,
                json={"model": self._model, "input": text},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            # /api/embed returns {"embeddings": [[...]]}
            if "embeddings" in data:
                return data["embeddings"][0]
            # /api/embeddings (old) returns {"embedding": [...]}
            if "embedding" in data:
                return data["embedding"]
            raise EmbeddingUnavailable(f"Unexpected Ollama response shape: {list(data.keys())}")
        except httpx.HTTPStatusError as e:
            raise EmbeddingUnavailable(f"Ollama HTTP error {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise EmbeddingUnavailable(f"Cannot reach Ollama at {self._base_url}: {e}") from e

    async def get_embedding(self, text: str) -> list[float]:
        if self._client is None:
            raise EmbeddingUnavailable("HTTP client not initialized")

        if self._working_path is not None:
            return await self._try_embed_path(self._working_path, text)

        # Auto-detect: try configured path first, then fallback
        paths = [self._default_embed_path]
        fallback = "/api/embeddings" if self._default_embed_path != "/api/embeddings" else "/api/embed"
        paths.append(fallback)

        last_error: Optional[Exception] = None
        for path in paths:
            try:
                result = await self._try_embed_path(path, text)
                self._working_path = path
                logger.info("Ollama embed path resolved to %s", path)
                return result
            except EmbeddingUnavailable as e:
                last_error = e
                continue

        raise EmbeddingUnavailable(f"No working Ollama embed path found. Last error: {last_error}")

    async def probe_dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        vec = await self.get_embedding("probe")
        self._dimension = len(vec)
        return self._dimension

    async def is_available(self, timeout: float = 5.0) -> Optional[bool]:
        """Returns True if reachable, False if unreachable, None if timed out."""
        if self._client is None:
            return False
        try:
            await asyncio.wait_for(self.get_embedding("ping"), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return None
        except EmbeddingUnavailable:
            return False
