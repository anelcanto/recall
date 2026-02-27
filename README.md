# recall

A personal semantic memory system. Store, search, and manage memories locally using vector embeddings.

Everything runs on your machine: FastAPI server, Qdrant vector database (Docker), and Ollama for local embeddings. Zero cost, full privacy.

## Install

### From PyPI

```bash
pip install recall-cli
```

### From Homebrew

```bash
brew tap anelcanto/recall-cli
brew install recall-cli
```

### From source

```bash
git clone https://github.com/anelcanto/recall.git
cd recall
./install.sh
```

Or the quick version:

```bash
make install
```

## Prerequisites

- **Docker** — runs Qdrant vector database
- **Ollama** — local embeddings (`brew install ollama && ollama pull nomic-embed-text`)
- **uv** — Python package manager (`brew install uv`)

## Quick start

```bash
# One-time setup (creates ~/.recall/.env, starts Qdrant)
recall init

# Start the API server (in a terminal, keep it running)
recall serve

# In another terminal
recall add "The quick brown fox" --tag test
recall search "fox"
recall list
recall status
```

## CLI

```
recall init                                                       # Set up config + start Qdrant
recall serve [--host 127.0.0.1] [--port 8100] [--no-qdrant]     # Start API server
recall add "text" --tag work --source cli [--dedupe-key "..."]
recall search "query" --top-k 10 [--no-text] [--output table|json]
recall ingest <file> [--format lines|jsonl] [--source name] [--auto-dedupe]
recall list [--limit 20] [--cursor ...] [--output table|json]
recall delete <id>
recall status
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RECALL_API_URL` | `http://127.0.0.1:8100` | API server URL |
| `RECALL_API_TOKEN` | (none) | Bearer token for auth |

## Architecture

```
recall CLI  -->  FastAPI server (:8100)  -->  Qdrant (Docker :6333)
                       |
                       v
                 Ollama (:11434)
                 nomic-embed-text
```

- **FastAPI** serves the HTTP API
- **Qdrant** stores vectors and payloads
- **Ollama** generates embeddings locally using `nomic-embed-text`
- **CLI** talks to the API over HTTP

User config lives in `~/.recall/.env`. Qdrant data persists in a Docker volume.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memory` | Store a memory |
| `POST` | `/search` | Semantic search |
| `POST` | `/ingest` | Batch import |
| `GET` | `/memories` | List with pagination |
| `DELETE` | `/memory/{id}` | Delete a memory |
| `GET` | `/health` | Service health check |

## Claude Code Plugin

recall ships as a Claude Code plugin — Claude can store and search your memories directly during conversations, with no manual CLI commands needed.

### Install

```bash
pip install recall-cli
```

Then add the plugin to your Claude Code project:

```bash
# From the recall repo directory (or any project where you want recall available)
claude mcp add recall -- recall-mcp
```

Or add it manually to `.mcp.json` at your project root:

```json
{
  "mcpServers": {
    "recall": {
      "type": "stdio",
      "command": "recall-mcp"
    }
  }
}
```

### Prerequisites

`recall serve` must be running before Claude can use the MCP tools:

```bash
recall serve
```

### Available MCP tools

| Tool | Description |
|------|-------------|
| `store_memory(text, tags?, source?, dedupe_key?)` | Store a new memory. Returns ID. |
| `search_memories(query, top_k?)` | Semantic search. Returns scored results. |
| `list_memories(limit?)` | List recent memories. |
| `delete_memory(memory_id)` | Delete a memory by ID. |
| `check_health()` | Check if recall API, Qdrant, and Ollama are up. |

### Usage examples

Once connected, just talk to Claude naturally:

```
"Remember that I prefer using uv for Python projects"
→ Claude calls store_memory(...)

"What do you know about my React setup?"
→ Claude calls search_memories("React setup")

"Show me all my memories"
→ Claude calls list_memories()

"Forget the last thing you stored"
→ Claude calls delete_memory(id)
```

### Check for updates

Use the `/recall:update` command in Claude Code to check for a newer version on PyPI and upgrade.

### Configuration

The MCP server reads config from `~/.recall/.env` (created by `recall init`) and respects the same environment variables as the CLI:

| Variable | Default | Description |
|----------|---------|-------------|
| `RECALL_API_URL` | `http://127.0.0.1:8100` | API server URL |
| `RECALL_API_TOKEN` | (none) | Bearer token for auth |

## Development

```bash
make test              # Unit tests (no services needed)
make test-integration  # Integration tests (Qdrant + Ollama required)
make test-degraded     # Degraded mode tests (Qdrant only)
make test-all          # All tests
```

## License

MIT
