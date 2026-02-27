#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="recall", help="Personal semantic memory CLI")
console = Console()

DEFAULT_API_URL = "http://127.0.0.1:8100"


def _get_api_url(api_url: Optional[str]) -> str:
    return api_url or os.environ.get("RECALL_API_URL", DEFAULT_API_URL)


def _get_token(token: Optional[str]) -> Optional[str]:
    return token or os.environ.get("RECALL_API_TOKEN") or None


def _build_headers(token: Optional[str]) -> dict:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _handle_error(resp: httpx.Response, api_url: str) -> None:
    if resp.status_code in (200, 201):
        return
    try:
        data = resp.json()
        detail = data.get("detail", resp.text)
    except Exception:
        detail = resp.text
    console.print(f"[red]Error {resp.status_code}:[/red] {detail}")
    raise typer.Exit(1)


def _connection_error(url: str) -> None:
    console.print(
        f"[red]Cannot reach recall service at {url}.[/red]\n"
        f"Try: [bold]make serve[/bold] (from your recall project directory)"
    )
    raise typer.Exit(1)


def _is_tty() -> bool:
    return sys.stdout.isatty()


@app.command()
def add(
    text: str = typer.Argument(..., help="The memory text to store"),
    tag: list[str] = typer.Option([], "--tag", "-t", help="Tag(s) to attach"),
    source: str = typer.Option("cli", "--source", "-s", help="Source identifier"),
    dedupe_key: Optional[str] = typer.Option(None, "--dedupe-key", "-d", help="Deduplication key"),
    api_url: Optional[str] = typer.Option(None, "--api-url"),
    token: Optional[str] = typer.Option(None, "--token"),
):
    """Store a memory."""
    url = _get_api_url(api_url)
    tok = _get_token(token)
    payload = {"text": text, "tags": list(tag), "source": source, "dedupe_key": dedupe_key}

    try:
        resp = httpx.post(f"{url}/memory", json=payload, headers=_build_headers(tok), timeout=30)
    except httpx.RequestError:
        _connection_error(url)

    _handle_error(resp, url)
    data = resp.json()
    console.print(f"[green]Stored[/green] id=[bold]{data['id']}[/bold] strategy={data['id_strategy']}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    no_text: bool = typer.Option(False, "--no-text"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="table|json"),
    api_url: Optional[str] = typer.Option(None, "--api-url"),
    token: Optional[str] = typer.Option(None, "--token"),
):
    """Search memories by semantic similarity."""
    url = _get_api_url(api_url)
    tok = _get_token(token)
    fmt = output or ("table" if _is_tty() else "json")
    payload = {"query": query, "top_k": top_k, "include_text": not no_text}

    try:
        resp = httpx.post(f"{url}/search", json=payload, headers=_build_headers(tok), timeout=30)
    except httpx.RequestError:
        _connection_error(url)

    _handle_error(resp, url)
    results = resp.json().get("results", [])

    if fmt == "json":
        print(json.dumps(results, indent=2))
        return

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Search: {query!r}")
    table.add_column("Score", style="cyan", width=6)
    table.add_column("ID", style="dim", width=36)
    table.add_column("Tags")
    table.add_column("Written At")
    if not no_text:
        table.add_column("Text")

    for r in results:
        row = [
            f"{r['score']:.3f}",
            r["id"],
            ", ".join(r.get("tags", [])),
            r.get("written_at", "")[:19],
        ]
        if not no_text:
            row.append((r.get("text") or "")[:80])
        table.add_row(*row)

    console.print(table)


@app.command()
def ingest(
    file: Path = typer.Argument(..., help="File to ingest"),
    format: str = typer.Option("lines", "--format", "-f", help="lines|jsonl"),
    source: str = typer.Option("ingest", "--source", "-s"),
    auto_dedupe: bool = typer.Option(False, "--auto-dedupe"),
    api_url: Optional[str] = typer.Option(None, "--api-url"),
    token: Optional[str] = typer.Option(None, "--token"),
):
    """Ingest memories from a file."""
    url = _get_api_url(api_url)
    tok = _get_token(token)

    if not file.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)

    raw = file.read_text(encoding="utf-8")
    items: list[dict] = []

    if format == "jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(obj)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            items.append({"text": line, "tags": [], "source": source})

    if auto_dedupe:
        for item in items:
            src = item.get("source", source)
            item["dedupe_key"] = hashlib.sha256(
                (item["text"] + src).encode()
            ).hexdigest()

    batch_size = 100
    total_succeeded = 0
    total_failed = 0
    all_errors: list[dict] = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        try:
            resp = httpx.post(
                f"{url}/ingest",
                json={"items": batch},
                headers=_build_headers(tok),
                timeout=60,
            )
        except httpx.RequestError:
            _connection_error(url)

        _handle_error(resp, url)
        data = resp.json()
        total_succeeded += data.get("succeeded", 0)
        total_failed += data.get("failed", 0)
        for err in data.get("errors", []):
            err["index"] += i
            all_errors.append(err)

    console.print(
        f"[green]Ingested[/green] {total_succeeded} succeeded, "
        f"[{'red' if total_failed else 'green'}]{total_failed}[/{'red' if total_failed else 'green'}] failed"
    )
    for err in all_errors:
        console.print(f"  [red]Error[/red] item {err['index']}: {err['error']}")


@app.command(name="list")
def list_memories(
    limit: int = typer.Option(20, "--limit", "-l"),
    cursor: Optional[str] = typer.Option(None, "--cursor"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="table|json"),
    api_url: Optional[str] = typer.Option(None, "--api-url"),
    token: Optional[str] = typer.Option(None, "--token"),
):
    """List stored memories."""
    url = _get_api_url(api_url)
    tok = _get_token(token)
    fmt = output or ("table" if _is_tty() else "json")

    params: dict = {"limit": limit}
    if cursor:
        params["cursor"] = cursor

    try:
        resp = httpx.get(f"{url}/memories", params=params, headers=_build_headers(tok), timeout=30)
    except httpx.RequestError:
        _connection_error(url)

    _handle_error(resp, url)
    data = resp.json()
    memories = data.get("memories", [])
    next_cursor = data.get("next_cursor")

    if fmt == "json":
        print(json.dumps(data, indent=2))
        return

    if not memories:
        console.print("[yellow]No memories found.[/yellow]")
        return

    table = Table(title="Memories")
    table.add_column("ID", style="dim", width=36)
    table.add_column("Tags")
    table.add_column("Source")
    table.add_column("Written At")
    table.add_column("Text")

    for m in memories:
        table.add_row(
            m["id"],
            ", ".join(m.get("tags", [])),
            m.get("source", ""),
            m.get("written_at", "")[:19],
            (m.get("text") or "")[:60],
        )

    console.print(table)
    if next_cursor:
        console.print(f"\n[dim]Next cursor:[/dim] {next_cursor}")


@app.command()
def delete(
    id: str = typer.Argument(..., help="Memory ID to delete"),
    api_url: Optional[str] = typer.Option(None, "--api-url"),
    token: Optional[str] = typer.Option(None, "--token"),
):
    """Delete a memory by ID."""
    url = _get_api_url(api_url)
    tok = _get_token(token)

    try:
        resp = httpx.delete(f"{url}/memory/{id}", headers=_build_headers(tok), timeout=30)
    except httpx.RequestError:
        _connection_error(url)

    _handle_error(resp, url)
    console.print(f"[green]Deleted[/green] {id}")


@app.command()
def status(
    api_url: Optional[str] = typer.Option(None, "--api-url"),
    token: Optional[str] = typer.Option(None, "--token"),
):
    """Show API health status."""
    url = _get_api_url(api_url)

    try:
        resp = httpx.get(f"{url}/health", timeout=10)
    except httpx.RequestError:
        _connection_error(url)

    data = resp.json()
    color = "green" if data.get("status") == "ok" else "yellow" if data.get("status") == "degraded" else "red"
    console.print(f"[{color}]Status:[/{color}] {data.get('status')}")
    console.print(f"  Qdrant: {data.get('qdrant')}")
    console.print(f"  Ollama: {data.get('ollama')}")


if __name__ == "__main__":
    app()
