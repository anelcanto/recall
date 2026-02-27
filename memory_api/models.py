from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from memory_api.config import settings


class MemoryCreate(BaseModel):
    text: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    source: str = Field(default="cli", max_length=200)
    dedupe_key: Optional[str] = None
    external_id: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_max_length(cls, v: str) -> str:
        if len(v) > settings.max_text_length:
            raise ValueError(
                f"text exceeds maximum length of {settings.max_text_length} characters"
            )
        return v

    @field_validator("tags")
    @classmethod
    def tags_limits(cls, v: list[str]) -> list[str]:
        if len(v) > 20:
            raise ValueError("too many tags (max 20)")
        for tag in v:
            if len(tag) > 100:
                raise ValueError(f"tag '{tag[:20]}...' exceeds 100 characters")
        return v


class MemoryCreateResponse(BaseModel):
    id: str
    id_strategy: str  # "random" | "deduped"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    include_text: bool = True

    @field_validator("query")
    @classmethod
    def query_max_length(cls, v: str) -> str:
        if len(v) > settings.max_text_length:
            raise ValueError(
                f"query exceeds maximum length of {settings.max_text_length} characters"
            )
        return v


class SearchResult(BaseModel):
    id: str
    score: float
    tags: list[str]
    source: str
    written_at: str
    text: Optional[str] = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class IngestItem(BaseModel):
    text: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    source: str = Field(default="cli", max_length=200)
    dedupe_key: Optional[str] = None

    @field_validator("tags")
    @classmethod
    def tags_limits(cls, v: list[str]) -> list[str]:
        if len(v) > 20:
            raise ValueError("too many tags (max 20)")
        for tag in v:
            if len(tag) > 100:
                raise ValueError("tag exceeds 100 characters")
        return v


class IngestRequest(BaseModel):
    items: list[IngestItem] = Field(..., min_length=1)

    @field_validator("items")
    @classmethod
    def items_max_batch(cls, v: list[IngestItem]) -> list[IngestItem]:
        if len(v) > settings.max_batch_size:
            raise ValueError(f"batch size exceeds maximum of {settings.max_batch_size}")
        return v


class IngestError(BaseModel):
    index: int
    error: str


class IngestResponse(BaseModel):
    succeeded: int
    failed: int
    errors: list[IngestError]


class MemoryRecord(BaseModel):
    id: str
    text: str
    tags: list[str]
    source: str
    written_at: str
    first_written_at: str
    dedupe_key: Optional[str] = None
    external_id: Optional[str] = None


class ListResponse(BaseModel):
    memories: list[MemoryRecord]
    next_cursor: Optional[str] = None


class DeleteResponse(BaseModel):
    status: str = "deleted"


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded" | "unavailable"
    qdrant: Optional[bool] = None
    ollama: Optional[bool] = None
