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
# Start services
make up        # Start Qdrant in Docker
make serve     # Start API server (keep this running)

# In another terminal
recall add "The quick brown fox" --tag test
recall search "fox"
recall list
recall status
```

## CLI

```
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

User config lives in `~/.memories/.env`. Qdrant data persists in a Docker volume.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/memory` | Store a memory |
| `POST` | `/search` | Semantic search |
| `POST` | `/ingest` | Batch import |
| `GET` | `/memories` | List with pagination |
| `DELETE` | `/memory/{id}` | Delete a memory |
| `GET` | `/health` | Service health check |

## Development

```bash
make test              # Unit tests (no services needed)
make test-integration  # Integration tests (Qdrant + Ollama required)
make test-degraded     # Degraded mode tests (Qdrant only)
make test-all          # All tests
```

## License

MIT
