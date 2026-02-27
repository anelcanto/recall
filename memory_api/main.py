from __future__ import annotations

import asyncio
import logging
import os
import secrets
import sys
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient

from memory_api.auth import require_auth
from memory_api.config import settings
from memory_api.embeddings import EmbeddingUnavailable, OllamaClient
from memory_api.models import (
    DeleteResponse,
    HealthResponse,
    IngestError,
    IngestRequest,
    IngestResponse,
    ListResponse,
    MemoryCreate,
    MemoryCreateResponse,
    MemoryRecord,
    SearchRequest,
    SearchResponse,
)
from memory_api.store import (
    MemoryStore,
    ModelMismatchError,
    QdrantCollectionNotFound,
    QdrantConnectionError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_store: Optional[MemoryStore] = None


def get_store() -> MemoryStore:
    assert _store is not None, "Store not initialized"
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store

    http_client = httpx.AsyncClient()

    embedder = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.embed_model,
        embed_path=settings.ollama_embed_path,
    )
    embedder.set_client(http_client)

    qdrant_client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        check_compatibility=False,
    )

    cursor_secret = settings.api_auth_token or secrets.token_hex(32)

    _store = MemoryStore(
        qdrant_host=settings.qdrant_host,
        qdrant_port=settings.qdrant_port,
        collection_name=settings.collection_name,
        embedder=embedder,
        cursor_secret=cursor_secret,
    )
    _store.set_client(qdrant_client)

    # Startup checks — warn but never crash (except model mismatch)
    try:
        collection_exists = await _store.collection_exists()
        logger.info("Qdrant is up. Collection '%s' exists: %s", settings.collection_name, collection_exists)

        if collection_exists:
            try:
                await _store.validate_model()
            except ModelMismatchError as e:
                logger.error("STARTUP FAILURE: %s", e)
                await http_client.aclose()
                await qdrant_client.close()
                sys.exit(1)
    except QdrantConnectionError as e:
        logger.warning("Qdrant is not reachable at startup: %s. Continuing anyway.", e)

    ollama_ok = await embedder.is_available(timeout=settings.health_check_timeout_s)
    if ollama_ok is True:
        logger.info("Ollama is up.")
    elif ollama_ok is False:
        logger.warning("Ollama is not reachable at startup. Embedding endpoints will return 503.")
    else:
        logger.warning("Ollama health check timed out at startup.")

    yield

    await http_client.aclose()
    await qdrant_client.close()


app = FastAPI(title="Personal Memory API", lifespan=lifespan)


# --- Global exception handlers ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_code = exc.detail if isinstance(exc.detail, str) else "http_error"
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error_code, "detail": str(exc.detail)},
    )


@app.exception_handler(EmbeddingUnavailable)
async def embedding_unavailable_handler(request: Request, exc: EmbeddingUnavailable):
    return JSONResponse(
        status_code=503,
        content={"error": "embedding_unavailable", "detail": str(exc)},
    )


@app.exception_handler(QdrantConnectionError)
async def qdrant_connection_handler(request: Request, exc: QdrantConnectionError):
    return JSONResponse(
        status_code=503,
        content={"error": "qdrant_unavailable", "detail": str(exc)},
    )


@app.exception_handler(QdrantCollectionNotFound)
async def qdrant_not_found_handler(request: Request, exc: QdrantCollectionNotFound):
    return JSONResponse(
        status_code=404,
        content={"error": "collection_not_found", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unexpected error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "Unexpected error"},
    )


# --- Health ---

@app.get("/health", response_model=HealthResponse)
async def health(store: MemoryStore = Depends(get_store)):
    timeout = settings.health_check_timeout_s

    async def check_qdrant() -> Optional[bool]:
        try:
            await asyncio.wait_for(store.collection_exists(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return None
        except Exception:
            return False

    qdrant_status, ollama_status = await asyncio.gather(
        check_qdrant(),
        store._embedder.is_available(timeout=timeout),
    )

    if qdrant_status is True and ollama_status is True:
        overall = "ok"
    elif qdrant_status is True:
        overall = "degraded"
    else:
        overall = "unavailable"

    status_code = 200 if overall == "ok" else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": overall, "qdrant": qdrant_status, "ollama": ollama_status},
    )


# --- Store memory ---

@app.post("/memory", dependencies=[Depends(require_auth)])
async def create_memory(
    body: MemoryCreate,
    store: MemoryStore = Depends(get_store),
):
    memory_id, id_strategy = await store.store_memory(
        text=body.text,
        tags=body.tags,
        source=body.source,
        dedupe_key=body.dedupe_key,
        external_id=body.external_id,
    )
    status_code = 201 if id_strategy == "random" else 200
    return JSONResponse(
        status_code=status_code,
        content={"id": memory_id, "id_strategy": id_strategy},
    )


# --- Search ---

@app.post("/search", dependencies=[Depends(require_auth)])
async def search_memories(
    body: SearchRequest,
    store: MemoryStore = Depends(get_store),
):
    results = await store.search(
        query=body.query,
        top_k=body.top_k,
        include_text=body.include_text,
    )
    return {"results": results}


# --- Ingest ---

@app.post("/ingest", dependencies=[Depends(require_auth)])
async def ingest_memories(
    body: IngestRequest,
    store: MemoryStore = Depends(get_store),
):
    succeeded = 0
    failed = 0
    errors: list[dict] = []

    for i, item in enumerate(body.items):
        if len(item.text) > settings.max_text_length:
            failed += 1
            errors.append(
                {
                    "index": i,
                    "error": f"text exceeds maximum length of {settings.max_text_length} characters",
                }
            )
            continue

        try:
            await store.store_memory(
                text=item.text,
                tags=item.tags,
                source=item.source,
                dedupe_key=item.dedupe_key,
                external_id=None,
            )
            succeeded += 1
        except EmbeddingUnavailable as e:
            failed += 1
            errors.append({"index": i, "error": f"embedding_unavailable: {e}"})
        except QdrantConnectionError as e:
            failed += 1
            errors.append({"index": i, "error": f"qdrant_unavailable: {e}"})
        except Exception as e:
            failed += 1
            errors.append({"index": i, "error": str(e)})

    return {"succeeded": succeeded, "failed": failed, "errors": errors}


# --- List ---

@app.get("/memories", dependencies=[Depends(require_auth)])
async def list_memories(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: Optional[str] = Query(default=None),
    store: MemoryStore = Depends(get_store),
):
    try:
        memories, next_cursor = await store.list_memories(limit=limit, cursor=cursor)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_cursor", "detail": "Cursor is invalid or tampered"},
        )

    return {"memories": memories, "next_cursor": next_cursor}


# --- Delete ---

@app.delete("/memory/{memory_id}", dependencies=[Depends(require_auth)])
async def delete_memory(
    memory_id: str,
    store: MemoryStore = Depends(get_store),
):
    await store.delete_memory(memory_id)
    return {"status": "deleted"}
