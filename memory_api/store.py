from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from memory_api.embeddings import EmbeddingUnavailable, OllamaClient

logger = logging.getLogger(__name__)

APP_NAMESPACE = uuid.UUID(
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
)  # DNS namespace reused as stable app UUID


class QdrantConnectionError(Exception):
    pass


class QdrantCollectionNotFound(Exception):
    pass


class ModelMismatchError(Exception):
    pass


class LRUCache:
    """Bounded LRU cache for asyncio locks."""

    def __init__(self, maxsize: int = 1000):
        self._maxsize = maxsize
        self._cache: OrderedDict[str, asyncio.Lock] = OrderedDict()

    def get_or_create(self, key: str) -> asyncio.Lock:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        lock = asyncio.Lock()
        self._cache[key] = lock
        self._cache.move_to_end(key)

        while len(self._cache) > self._maxsize:
            oldest_key, oldest_lock = next(iter(self._cache.items()))
            if not oldest_lock.locked():
                del self._cache[oldest_key]
            else:
                # Skip locked entries — move to end and stop evicting to avoid starvation
                self._cache.move_to_end(oldest_key)
                break

        return lock


class MemoryStore:
    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        embedder: OllamaClient,
        cursor_secret: str,
    ):
        self._host = qdrant_host
        self._port = qdrant_port
        self._collection_name = collection_name
        self._embedder = embedder
        self._cursor_secret = cursor_secret
        self._client: Optional[AsyncQdrantClient] = None

        self._collection_exists_cache: Optional[bool] = None
        self._collection_exists_cached_at: float = 0.0
        self._collection_cache_ttl: float = 30.0

        self._dedupe_locks: LRUCache = LRUCache(maxsize=1000)

    def set_client(self, client: AsyncQdrantClient) -> None:
        self._client = client

    def _invalidate_collection_cache(self) -> None:
        self._collection_exists_cache = None
        self._collection_exists_cached_at = 0.0

    async def collection_exists(self) -> bool:
        now = time.monotonic()
        if (
            self._collection_exists_cache is True
            and (now - self._collection_exists_cached_at) < self._collection_cache_ttl
        ):
            return True

        try:
            assert self._client is not None
            result = await self._client.collection_exists(self._collection_name)
            if result:
                self._collection_exists_cache = True
                self._collection_exists_cached_at = now
            else:
                self._collection_exists_cache = False
            return result
        except Exception as e:
            self._invalidate_collection_cache()
            raise QdrantConnectionError(f"Cannot reach Qdrant: {e}") from e

    async def ensure_collection(self) -> None:
        if await self.collection_exists():
            return

        dim = await self._embedder.probe_dimension()
        assert self._client is not None

        try:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qmodels.VectorParams(
                    size=dim,
                    distance=qmodels.Distance.COSINE,
                ),
                on_disk_payload=False,
            )
            await self._client.update_collection(
                collection_name=self._collection_name,
                optimizers_config=qmodels.OptimizersConfigDiff(),
            )
            # Store description in a dedicated point with id=0 approach is fragile;
            # instead use collection info's description field via REST if available,
            # or store as a known metadata point.
            # Qdrant doesn't have a free-form description field on the collection itself
            # in the Python client API, so we store model metadata in a sentinel payload point.
            await self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    qmodels.PointStruct(
                        id=str(uuid.uuid5(APP_NAMESPACE, "__meta__")),
                        vector=[0.0] * dim,
                        payload={
                            "schema_version": 1,
                            "_meta": True,
                            "model": self._embedder._model,
                            "dim": dim,
                        },
                    )
                ],
            )
        except Exception as e:
            self._invalidate_collection_cache()
            raise QdrantConnectionError(f"Failed to create collection: {e}") from e

        await self._create_payload_indexes()

        self._collection_exists_cache = True
        self._collection_exists_cached_at = time.monotonic()

    async def _create_payload_indexes(self) -> None:
        assert self._client is not None
        indexes = [
            ("dedupe_key", qmodels.PayloadSchemaType.KEYWORD),
            ("tags", qmodels.PayloadSchemaType.KEYWORD),
            ("source", qmodels.PayloadSchemaType.KEYWORD),
        ]
        for field, schema in indexes:
            try:
                await self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as e:
                logger.warning("Could not create index for %s: %s", field, e)

        # written_at as datetime index
        try:
            await self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name="written_at",
                field_schema=qmodels.PayloadSchemaType.DATETIME,
            )
        except Exception as e:
            logger.warning("Could not create written_at index: %s", e)

    async def validate_model(self) -> None:
        """Check collection's stored model matches current config. Raises ModelMismatchError on mismatch."""
        assert self._client is not None
        meta_id = str(uuid.uuid5(APP_NAMESPACE, "__meta__"))
        try:
            results = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=[meta_id],
                with_payload=True,
            )
        except Exception as e:
            raise QdrantConnectionError(f"Cannot read collection metadata: {e}") from e

        if not results:
            logger.warning(
                "Collection '%s' has no model metadata (pre-v1 or external). Proceeding.",
                self._collection_name,
            )
            return

        meta = results[0].payload or {}
        if not meta.get("_meta"):
            logger.warning("Collection metadata point found but _meta flag missing. Proceeding.")
            return

        stored_model = meta.get("model")
        stored_dim = meta.get("dim")

        if stored_model is None:
            logger.warning("No model stored in metadata. Proceeding.")
            return

        current_model = self._embedder._model
        try:
            current_dim = await self._embedder.probe_dimension()
        except EmbeddingUnavailable:
            logger.warning(
                "Cannot probe Ollama dimension for model validation. Skipping validation."
            )
            return

        if stored_model != current_model or stored_dim != current_dim:
            raise ModelMismatchError(
                f"Model mismatch: collection uses {stored_model} ({stored_dim}) "
                f"but EMBED_MODEL={current_model} ({current_dim}). "
                f"Delete collection or change EMBED_MODEL."
            )

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    async def store_memory(
        self,
        text: str,
        tags: list[str],
        source: str,
        dedupe_key: Optional[str],
        external_id: Optional[str],
    ) -> tuple[str, str]:
        """Returns (id, id_strategy). id_strategy is 'random' or 'deduped'."""
        await self.ensure_collection()

        now = self._now_iso()

        if dedupe_key is None:
            point_id = str(uuid.uuid4())
            vector = await self._embedder.get_embedding(text)
            await self._upsert_point(
                point_id=point_id,
                vector=vector,
                payload={
                    "schema_version": 1,
                    "text": text,
                    "tags": tags,
                    "source": source,
                    "dedupe_key": None,
                    "external_id": external_id,
                    "written_at": now,
                    "first_written_at": now,
                },
            )
            return point_id, "random"

        # Deduped write with per-key lock
        point_id = str(uuid.uuid5(APP_NAMESPACE, f"v1:{dedupe_key}"))
        lock = self._dedupe_locks.get_or_create(dedupe_key)

        async with lock:
            # Preserve first_written_at if point already exists
            first_written_at = now
            assert self._client is not None
            try:
                existing = await self._client.retrieve(
                    collection_name=self._collection_name,
                    ids=[point_id],
                    with_payload=True,
                )
                if existing and existing[0].payload:
                    first_written_at = existing[0].payload.get("first_written_at", now)
            except Exception:
                pass  # If we can't read, treat as new

            vector = await self._embedder.get_embedding(text)
            await self._upsert_point(
                point_id=point_id,
                vector=vector,
                payload={
                    "schema_version": 1,
                    "text": text,
                    "tags": tags,
                    "source": source,
                    "dedupe_key": dedupe_key,
                    "external_id": external_id,
                    "written_at": now,
                    "first_written_at": first_written_at,
                },
            )

        return point_id, "deduped"

    async def _upsert_point(self, point_id: str, vector: list[float], payload: dict) -> None:
        assert self._client is not None
        try:
            await self._client.upsert(
                collection_name=self._collection_name,
                points=[
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
        except Exception as e:
            raise QdrantConnectionError(f"Failed to upsert point: {e}") from e

    async def search(
        self,
        query: str,
        top_k: int,
        include_text: bool,
    ) -> list[dict]:
        try:
            exists = await self.collection_exists()
        except QdrantConnectionError:
            raise
        if not exists:
            return []

        vector = await self._embedder.get_embedding(query)
        assert self._client is not None

        try:
            results = await self._client.search(
                collection_name=self._collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
                query_filter=qmodels.Filter(
                    must_not=[
                        qmodels.FieldCondition(
                            key="_meta",
                            match=qmodels.MatchValue(value=True),
                        )
                    ]
                ),
            )
        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                self._invalidate_collection_cache()
                return []
            raise QdrantConnectionError(f"Qdrant search error: {e}") from e
        except Exception as e:
            raise QdrantConnectionError(f"Qdrant search error: {e}") from e

        output = []
        for r in results:
            p = r.payload or {}
            item: dict = {
                "id": str(r.id),
                "score": r.score,
                "tags": p.get("tags", []),
                "source": p.get("source", ""),
                "written_at": p.get("written_at", ""),
            }
            if include_text:
                item["text"] = p.get("text")
            output.append(item)
        return output

    async def list_memories(
        self, limit: int, cursor: Optional[str]
    ) -> tuple[list[dict], Optional[str]]:
        try:
            exists = await self.collection_exists()
        except QdrantConnectionError:
            raise
        if not exists:
            return [], None

        offset = None
        if cursor is not None:
            offset = self._decode_cursor(cursor)

        assert self._client is not None
        try:
            result = await self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=qmodels.Filter(
                    must_not=[
                        qmodels.FieldCondition(
                            key="_meta",
                            match=qmodels.MatchValue(value=True),
                        )
                    ]
                ),
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                order_by=qmodels.OrderBy(
                    key="written_at",
                    direction=qmodels.Direction.Desc,
                ),
            )
        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                self._invalidate_collection_cache()
                return [], None
            raise QdrantConnectionError(f"Qdrant list error: {e}") from e
        except Exception as e:
            raise QdrantConnectionError(f"Qdrant list error: {e}") from e

        points, next_page_offset = result

        memories = []
        for p in points:
            payload = p.payload or {}
            memories.append(
                {
                    "id": str(p.id),
                    "text": payload.get("text", ""),
                    "tags": payload.get("tags", []),
                    "source": payload.get("source", ""),
                    "written_at": payload.get("written_at", ""),
                    "first_written_at": payload.get("first_written_at", ""),
                    "dedupe_key": payload.get("dedupe_key"),
                    "external_id": payload.get("external_id"),
                }
            )

        next_cursor = None
        if next_page_offset is not None:
            next_cursor = self._encode_cursor(next_page_offset)

        return memories, next_cursor

    async def delete_memory(self, memory_id: str) -> None:
        try:
            exists = await self.collection_exists()
        except QdrantConnectionError:
            raise
        if not exists:
            raise QdrantCollectionNotFound(f"Memory {memory_id} not found")

        assert self._client is not None
        try:
            # Check if point exists first
            points = await self._client.retrieve(
                collection_name=self._collection_name,
                ids=[memory_id],
            )
        except UnexpectedResponse as e:
            if "not found" in str(e).lower():
                self._invalidate_collection_cache()
                raise QdrantCollectionNotFound(f"Memory {memory_id} not found")
            raise QdrantConnectionError(f"Qdrant error: {e}") from e
        except Exception as e:
            raise QdrantConnectionError(f"Qdrant error: {e}") from e

        if not points:
            raise QdrantCollectionNotFound(f"Memory {memory_id} not found")

        try:
            await self._client.delete(
                collection_name=self._collection_name,
                points_selector=qmodels.PointIdsList(points=[memory_id]),
            )
        except Exception as e:
            raise QdrantConnectionError(f"Failed to delete point: {e}") from e

    def _encode_cursor(self, offset: Any) -> str:
        data = json.dumps({"offset": offset})
        sig = hmac.new(
            self._cursor_secret.encode(),
            data.encode(),
            hashlib.sha256,
        ).hexdigest()
        payload = json.dumps({"offset": offset, "qh": sig})
        return base64.urlsafe_b64encode(payload.encode()).decode()

    def _decode_cursor(self, cursor: str) -> Any:
        try:
            raw = base64.urlsafe_b64decode(cursor.encode()).decode()
            obj = json.loads(raw)
            offset = obj["offset"]
            qh = obj["qh"]
        except Exception:
            raise ValueError("invalid_cursor")

        data = json.dumps({"offset": offset})
        expected = hmac.new(
            self._cursor_secret.encode(),
            data.encode(),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(qh, expected):
            raise ValueError("invalid_cursor")

        return offset
