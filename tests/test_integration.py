"""Integration tests — requires Qdrant + Ollama running."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import uuid
from typing import AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# We override collection name per test via unique fixture
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


@pytest_asyncio.fixture
async def unique_collection():
    """Yield a unique collection name and clean up after the test."""
    name = f"memories_test_{uuid.uuid4().hex}"
    yield name
    # Cleanup
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    try:
        exists = await client.collection_exists(name)
        if exists:
            await client.delete_collection(name)
    finally:
        await client.close()


@pytest_asyncio.fixture
async def app_client(unique_collection) -> AsyncGenerator:
    """Build a TestClient with a fresh app using the unique collection."""
    import os

    os.environ["COLLECTION_NAME"] = unique_collection
    os.environ["API_AUTH_TOKEN"] = ""

    # Re-import to get fresh settings
    import importlib

    import memory_api.config as cfg_mod

    importlib.reload(cfg_mod)

    import memory_api.main as main_mod

    importlib.reload(main_mod)

    from memory_api.main import app

    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        # Trigger lifespan via ASGI
        async with app.router.lifespan_context(app):
            yield client


@pytest.fixture
def sync_client(unique_collection):
    """Synchronous test client using requests-style approach."""
    import os

    os.environ["COLLECTION_NAME"] = unique_collection
    os.environ["API_AUTH_TOKEN"] = ""

    import importlib

    import memory_api.config as cfg_mod

    importlib.reload(cfg_mod)

    import memory_api.main as main_mod

    importlib.reload(main_mod)

    from memory_api.main import app
    from starlette.testclient import TestClient as StarletteClient

    with StarletteClient(app=app, raise_server_exceptions=False) as client:
        yield client


# Helper to add a memory via test client
def add_memory(client, text: str, tags=None, source="test", dedupe_key=None):
    payload = {"text": text, "tags": tags or [], "source": source}
    if dedupe_key is not None:
        payload["dedupe_key"] = dedupe_key
    return client.post("/memory", json=payload)


class TestHealth:
    def test_health_ok(self, sync_client):
        resp = sync_client.get("/health")
        data = resp.json()
        assert data["status"] in ("ok", "degraded")
        assert data["qdrant"] is True


class TestStoreAndSearch:
    def test_store_returns_201(self, sync_client):
        resp = add_memory(sync_client, "Hello world memory")
        assert resp.status_code == 201
        data = resp.json()
        assert "id" in data
        assert data["id_strategy"] == "random"

    def test_search_roundtrip(self, sync_client):
        add_memory(sync_client, "The quick brown fox")
        resp = sync_client.post("/search", json={"query": "fox", "top_k": 5})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) >= 1
        texts = [r["text"] for r in results]
        assert any("fox" in (t or "") for t in texts)


class TestDedupe:
    def test_deduped_same_key_one_point(self, sync_client):
        key = "my-unique-key"
        r1 = add_memory(sync_client, "First write", dedupe_key=key)
        r2 = add_memory(sync_client, "Second write", dedupe_key=key)
        assert r1.status_code == 201
        assert r2.status_code == 200
        assert r1.json()["id"] == r2.json()["id"]
        assert r1.json()["id_strategy"] == "random"  # first write = new
        assert r2.json()["id_strategy"] == "deduped"

    def test_deduped_preserves_first_written_at(self, sync_client):
        key = "preserve-key"
        add_memory(sync_client, "First", dedupe_key=key)
        add_memory(sync_client, "Second", dedupe_key=key)

        resp = sync_client.get("/memories", params={"limit": 10})
        memories = resp.json()["memories"]
        mem = next((m for m in memories if m.get("dedupe_key") == key), None)
        assert mem is not None
        assert mem["first_written_at"] <= mem["written_at"]

    def test_no_dedupe_two_points(self, sync_client):
        add_memory(sync_client, "No dedupe A")
        add_memory(sync_client, "No dedupe B")
        resp = sync_client.get("/memories", params={"limit": 10})
        memories = resp.json()["memories"]
        assert len(memories) >= 2


class TestListPagination:
    def test_pagination_follows_cursors(self, sync_client):
        for i in range(5):
            add_memory(sync_client, f"Pagination memory {i}")

        resp = sync_client.get("/memories", params={"limit": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["memories"]) == 2
        assert data["next_cursor"] is not None

        # Follow cursor
        resp2 = sync_client.get("/memories", params={"limit": 2, "cursor": data["next_cursor"]})
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["memories"]) == 2

        # IDs should not overlap
        ids1 = {m["id"] for m in data["memories"]}
        ids2 = {m["id"] for m in data2["memories"]}
        assert not ids1.intersection(ids2)


class TestDelete:
    def test_delete_success(self, sync_client):
        resp = add_memory(sync_client, "To be deleted")
        memory_id = resp.json()["id"]

        del_resp = sync_client.delete(f"/memory/{memory_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

    def test_delete_not_found(self, sync_client):
        fake_id = str(uuid.uuid4())
        resp = sync_client.delete(f"/memory/{fake_id}")
        assert resp.status_code == 404

    def test_re_delete_404(self, sync_client):
        resp = add_memory(sync_client, "Delete twice")
        memory_id = resp.json()["id"]
        sync_client.delete(f"/memory/{memory_id}")
        resp2 = sync_client.delete(f"/memory/{memory_id}")
        assert resp2.status_code == 404


class TestIngest:
    def test_ingest_batch(self, sync_client):
        items = [{"text": f"Ingest item {i}", "tags": [], "source": "test"} for i in range(5)]
        resp = sync_client.post("/ingest", json={"items": items})
        assert resp.status_code == 200
        data = resp.json()
        assert data["succeeded"] == 5
        assert data["failed"] == 0

    def test_ingest_partial_failure(self, sync_client):
        items = [
            {"text": "Valid item", "tags": [], "source": "test"},
            {"text": "x" * 9000, "tags": [], "source": "test"},  # too long
            {"text": "Another valid", "tags": [], "source": "test"},
        ]
        resp = sync_client.post("/ingest", json={"items": items})
        assert resp.status_code == 200
        data = resp.json()
        assert data["succeeded"] == 2
        assert data["failed"] == 1
        assert len(data["errors"]) == 1
        assert data["errors"][0]["index"] == 1


class TestSizeLimits:
    def test_text_too_long_422(self, sync_client):
        resp = add_memory(sync_client, "x" * 8001)
        assert resp.status_code == 422

    def test_top_k_51_422(self, sync_client):
        add_memory(sync_client, "Something")
        resp = sync_client.post("/search", json={"query": "q", "top_k": 51})
        assert resp.status_code == 422


class TestIncludeText:
    def test_include_text_false(self, sync_client):
        add_memory(sync_client, "Secret text here")
        resp = sync_client.post("/search", json={"query": "secret", "top_k": 5, "include_text": False})
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert "text" not in r or r["text"] is None


class TestEmptyCollection:
    def test_list_empty(self, sync_client):
        resp = sync_client.get("/memories", params={"limit": 10})
        assert resp.status_code == 200
        assert resp.json()["memories"] == []

    def test_search_empty_no_embed(self, sync_client):
        # With no collection, search should return empty without calling Ollama
        resp = sync_client.post("/search", json={"query": "anything"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []


class TestAuth:
    def test_no_auth_required_when_no_token(self, sync_client):
        resp = add_memory(sync_client, "Auth test")
        assert resp.status_code == 201

    def test_health_no_auth_needed(self, sync_client):
        resp = sync_client.get("/health")
        assert resp.status_code in (200, 503)

    def test_auth_enforced_with_token(self, unique_collection):
        import os

        os.environ["COLLECTION_NAME"] = unique_collection
        os.environ["API_AUTH_TOKEN"] = "secret-token"

        import importlib

        import memory_api.config as cfg_mod

        importlib.reload(cfg_mod)

        import memory_api.main as main_mod

        importlib.reload(main_mod)

        from memory_api.main import app
        from starlette.testclient import TestClient

        with TestClient(app=app, raise_server_exceptions=False) as client:
            # Without token → 401
            resp = client.post("/memory", json={"text": "test"})
            assert resp.status_code == 401

            # With token → 201
            resp = client.post(
                "/memory",
                json={"text": "test"},
                headers={"Authorization": "Bearer secret-token"},
            )
            assert resp.status_code == 201

            # Health still works without token
            resp = client.get("/health")
            assert resp.status_code in (200, 503)

        os.environ["API_AUTH_TOKEN"] = ""


class TestCursorTampering:
    def test_tampered_cursor_400(self, sync_client):
        for i in range(3):
            add_memory(sync_client, f"Cursor test {i}")

        resp = sync_client.get("/memories", params={"limit": 1})
        cursor = resp.json().get("next_cursor")
        if cursor is None:
            pytest.skip("Not enough records to get a cursor")

        # Tamper with cursor: decode, change qh, re-encode
        raw = base64.urlsafe_b64decode(cursor.encode()).decode()
        obj = json.loads(raw)
        obj["qh"] = "deadbeef" * 8
        tampered = base64.urlsafe_b64encode(json.dumps(obj).encode()).decode()

        resp2 = sync_client.get("/memories", params={"limit": 1, "cursor": tampered})
        assert resp2.status_code == 400
        assert resp2.json()["error"] == "invalid_cursor"


class TestCollectionDrop:
    def test_cache_invalidates_on_drop(self, sync_client, unique_collection):
        add_memory(sync_client, "Will be dropped")

        # Drop collection externally
        from qdrant_client import QdrantClient

        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        qc.delete_collection(unique_collection)
        qc.close()

        # List should return empty (cache invalidates)
        resp = sync_client.get("/memories", params={"limit": 10})
        assert resp.status_code == 200
        assert resp.json()["memories"] == []
