"""Unit tests — no services required."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import uuid

import pytest

# ---- Validation ----
from memory_api.models import IngestRequest, MemoryCreate, SearchRequest

MAX_TEXT = 8000


class TestValidation:
    def test_text_at_limit(self):
        m = MemoryCreate(text="x" * MAX_TEXT)
        assert len(m.text) == MAX_TEXT

    def test_text_over_limit(self):
        with pytest.raises(Exception, match="exceeds maximum length"):
            MemoryCreate(text="x" * (MAX_TEXT + 1))

    def test_text_empty_rejected(self):
        with pytest.raises(Exception):
            MemoryCreate(text="")

    def test_tags_max(self):
        m = MemoryCreate(text="hi", tags=["t"] * 20)
        assert len(m.tags) == 20

    def test_tags_over_limit(self):
        with pytest.raises(Exception, match="too many tags"):
            MemoryCreate(text="hi", tags=["t"] * 21)

    def test_tag_too_long(self):
        with pytest.raises(Exception):
            MemoryCreate(text="hi", tags=["x" * 101])

    def test_tag_at_limit(self):
        m = MemoryCreate(text="hi", tags=["x" * 100])
        assert m.tags[0] == "x" * 100

    def test_batch_max(self):
        from memory_api.models import IngestItem

        items = [IngestItem(text="t") for _ in range(100)]
        req = IngestRequest(items=items)
        assert len(req.items) == 100

    def test_batch_over_limit(self):
        from memory_api.models import IngestItem

        items = [IngestItem(text="t") for _ in range(101)]
        with pytest.raises(Exception, match="batch size exceeds"):
            IngestRequest(items=items)

    def test_top_k_min(self):
        r = SearchRequest(query="q", top_k=1)
        assert r.top_k == 1

    def test_top_k_max(self):
        r = SearchRequest(query="q", top_k=50)
        assert r.top_k == 50

    def test_top_k_over_limit(self):
        with pytest.raises(Exception):
            SearchRequest(query="q", top_k=51)

    def test_top_k_under_limit(self):
        with pytest.raises(Exception):
            SearchRequest(query="q", top_k=0)

    def test_query_at_limit(self):
        r = SearchRequest(query="x" * MAX_TEXT)
        assert len(r.query) == MAX_TEXT

    def test_query_over_limit(self):
        with pytest.raises(Exception, match="exceeds maximum length"):
            SearchRequest(query="x" * (MAX_TEXT + 1))

    def test_source_max_length(self):
        m = MemoryCreate(text="hi", source="s" * 200)
        assert len(m.source) == 200


# ---- UUID v5 determinism ----

APP_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class TestUUIDv5:
    def test_same_key_same_id(self):
        id1 = str(uuid.uuid5(APP_NAMESPACE, "v1:my-key"))
        id2 = str(uuid.uuid5(APP_NAMESPACE, "v1:my-key"))
        assert id1 == id2

    def test_different_keys_different_ids(self):
        id1 = str(uuid.uuid5(APP_NAMESPACE, "v1:key-a"))
        id2 = str(uuid.uuid5(APP_NAMESPACE, "v1:key-b"))
        assert id1 != id2

    def test_v1_prefix_applied(self):
        id_with_prefix = str(uuid.uuid5(APP_NAMESPACE, "v1:abc"))
        id_without_prefix = str(uuid.uuid5(APP_NAMESPACE, "abc"))
        assert id_with_prefix != id_without_prefix

    def test_uuid_format(self):
        result = str(uuid.uuid5(APP_NAMESPACE, "v1:test"))
        # Validate it's a valid UUID string
        parsed = uuid.UUID(result)
        assert parsed.version == 5


# ---- Cursor encode/decode ----

SECRET = "test-secret-key"


def _encode_cursor(offset, secret: str = SECRET) -> str:
    data = json.dumps({"offset": offset})
    sig = hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()
    payload = json.dumps({"offset": offset, "qh": sig})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _decode_cursor(cursor: str, secret: str = SECRET):
    raw = base64.urlsafe_b64decode(cursor.encode()).decode()
    obj = json.loads(raw)
    offset = obj["offset"]
    qh = obj["qh"]
    data = json.dumps({"offset": offset})
    expected = hmac.new(secret.encode(), data.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(qh, expected):
        raise ValueError("invalid_cursor")
    return offset


class TestCursor:
    def test_roundtrip_int(self):
        encoded = _encode_cursor(42)
        decoded = _decode_cursor(encoded)
        assert decoded == 42

    def test_roundtrip_none(self):
        encoded = _encode_cursor(None)
        decoded = _decode_cursor(encoded)
        assert decoded is None

    def test_roundtrip_string_offset(self):
        encoded = _encode_cursor("some-uuid-offset")
        decoded = _decode_cursor(encoded)
        assert decoded == "some-uuid-offset"

    def test_hmac_mismatch_raises(self):
        encoded = _encode_cursor(10)
        # Tamper: decode, change qh, re-encode
        raw = base64.urlsafe_b64decode(encoded.encode()).decode()
        obj = json.loads(raw)
        obj["qh"] = "deadbeef" * 8
        tampered = base64.urlsafe_b64encode(json.dumps(obj).encode()).decode()
        with pytest.raises(ValueError, match="invalid_cursor"):
            _decode_cursor(tampered)

    def test_corrupted_base64_raises(self):
        with pytest.raises(Exception):
            _decode_cursor("not-valid-base64!!!")

    def test_different_secret_raises(self):
        encoded = _encode_cursor(5, secret="secret-a")
        with pytest.raises(ValueError, match="invalid_cursor"):
            _decode_cursor(encoded, secret="secret-b")


# ---- Config precedence ----


class TestConfigPrecedence:
    def test_env_default(self):
        from memory_api.config import Settings

        s = Settings()
        assert s.qdrant_host == "localhost"
        assert s.api_port == 8100
        assert s.max_text_length == 8000
        assert s.health_check_timeout_s == 5.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("QDRANT_HOST", "myhost")
        monkeypatch.setenv("API_PORT", "9999")
        from memory_api.config import Settings

        s = Settings()
        assert s.qdrant_host == "myhost"
        assert s.api_port == 9999
