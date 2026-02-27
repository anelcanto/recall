"""Unit tests for recall CLI commands (no services required)."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
from typer.testing import CliRunner

from cli.recall_cli import _build_headers, _get_api_url, app

runner = CliRunner()


def make_response(status_code: int, body) -> httpx.Response:
    content = json.dumps(body).encode()
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers={"Content-Type": "application/json"},
    )


class TestGetApiUrl:
    def test_get_api_url_default(self, monkeypatch):
        monkeypatch.delenv("RECALL_API_URL", raising=False)
        assert _get_api_url(None) == "http://127.0.0.1:8100"

    def test_get_api_url_env(self, monkeypatch):
        monkeypatch.setenv("RECALL_API_URL", "http://example.com:9000")
        assert _get_api_url(None) == "http://example.com:9000"


class TestBuildHeaders:
    def test_build_headers_no_token(self):
        h = _build_headers(None)
        assert h == {"Content-Type": "application/json"}

    def test_build_headers_with_token(self):
        h = _build_headers("mytoken")
        assert h["Authorization"] == "Bearer mytoken"
        assert h["Content-Type"] == "application/json"


class TestAdd:
    def test_add_success(self):
        fake_resp = make_response(200, {"id": "abc-123", "id_strategy": "uuid"})
        with patch("httpx.post", return_value=fake_resp):
            result = runner.invoke(
                app, ["add", "Hello world", "--api-url", "http://localhost:8100"]
            )
        assert result.exit_code == 0
        assert "Stored" in result.output
        assert "abc-123" in result.output

    def test_add_connection_error(self):
        with patch("httpx.post", side_effect=httpx.RequestError("connection refused")):
            result = runner.invoke(app, ["add", "Hello", "--api-url", "http://localhost:8100"])
        assert result.exit_code != 0
        assert "Cannot reach" in result.output


class TestSearch:
    def test_search_table_output(self):
        body = {
            "results": [
                {
                    "id": "id1",
                    "score": 0.95,
                    "text": "some text",
                    "tags": ["foo"],
                    "written_at": "2024-01-01T00:00:00",
                }
            ]
        }
        fake_resp = make_response(200, body)
        with patch("httpx.post", return_value=fake_resp):
            result = runner.invoke(
                app, ["search", "query", "--output", "table", "--api-url", "http://localhost:8100"]
            )
        assert result.exit_code == 0
        assert "id1" in result.output

    def test_search_json_output(self):
        body = {
            "results": [
                {
                    "id": "id1",
                    "score": 0.95,
                    "text": "some text",
                    "tags": [],
                    "written_at": "2024-01-01T00:00:00",
                }
            ]
        }
        fake_resp = make_response(200, body)
        with patch("httpx.post", return_value=fake_resp):
            result = runner.invoke(
                app, ["search", "query", "--output", "json", "--api-url", "http://localhost:8100"]
            )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed[0]["id"] == "id1"

    def test_search_no_results(self):
        fake_resp = make_response(200, {"results": []})
        with patch("httpx.post", return_value=fake_resp):
            result = runner.invoke(
                app, ["search", "query", "--output", "table", "--api-url", "http://localhost:8100"]
            )
        assert result.exit_code == 0
        assert "No results" in result.output


class TestList:
    def test_list_success(self):
        body = {
            "memories": [
                {
                    "id": "mem1",
                    "tags": ["t1"],
                    "source": "cli",
                    "written_at": "2024-01-01T00:00:00",
                    "text": "hello",
                }
            ],
            "next_cursor": None,
        }
        fake_resp = make_response(200, body)
        with patch("httpx.get", return_value=fake_resp):
            result = runner.invoke(
                app, ["list", "--output", "table", "--api-url", "http://localhost:8100"]
            )
        assert result.exit_code == 0
        assert "mem1" in result.output

    def test_list_json_output(self):
        body = {
            "memories": [
                {
                    "id": "mem1",
                    "tags": [],
                    "source": "cli",
                    "written_at": "2024-01-01T00:00:00",
                    "text": "hello",
                }
            ],
            "next_cursor": None,
        }
        fake_resp = make_response(200, body)
        with patch("httpx.get", return_value=fake_resp):
            result = runner.invoke(
                app, ["list", "--output", "json", "--api-url", "http://localhost:8100"]
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["memories"][0]["id"] == "mem1"

    def test_list_with_cursor(self):
        body = {"memories": [], "next_cursor": None}
        fake_resp = make_response(200, body)
        with patch("httpx.get", return_value=fake_resp) as mock_get:
            runner.invoke(
                app,
                [
                    "list",
                    "--cursor",
                    "abc123",
                    "--output",
                    "table",
                    "--api-url",
                    "http://localhost:8100",
                ],
            )
        call_kwargs = mock_get.call_args[1]
        assert "cursor" in call_kwargs.get("params", {})
        assert call_kwargs["params"]["cursor"] == "abc123"


class TestDelete:
    def test_delete_success(self):
        fake_resp = make_response(200, {"message": "Deleted"})
        with patch("httpx.delete", return_value=fake_resp):
            result = runner.invoke(app, ["delete", "abc-123", "--api-url", "http://localhost:8100"])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert "abc-123" in result.output

    def test_delete_error_404(self):
        fake_resp = make_response(404, {"detail": "Memory not found"})
        with patch("httpx.delete", return_value=fake_resp):
            result = runner.invoke(app, ["delete", "bad-id", "--api-url", "http://localhost:8100"])
        assert result.exit_code != 0
        assert "404" in result.output


class TestStatus:
    def test_status_ok(self):
        body = {"status": "ok", "qdrant": "healthy", "ollama": "healthy"}
        fake_resp = make_response(200, body)
        with patch("httpx.get", return_value=fake_resp):
            result = runner.invoke(app, ["status", "--api-url", "http://localhost:8100"])
        assert result.exit_code == 0
        assert "ok" in result.output

    def test_status_degraded(self):
        body = {"status": "degraded", "qdrant": "healthy", "ollama": "unavailable"}
        fake_resp = make_response(200, body)
        with patch("httpx.get", return_value=fake_resp):
            result = runner.invoke(app, ["status", "--api-url", "http://localhost:8100"])
        assert result.exit_code == 0
        assert "degraded" in result.output
