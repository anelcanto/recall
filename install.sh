#!/usr/bin/env bash
set -euo pipefail

echo "=== recall installer ==="
echo ""

# Check prerequisites
missing=()
command -v docker &>/dev/null || missing+=("docker")
command -v uv &>/dev/null || missing+=("uv (brew install uv)")

if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing prerequisites:"
    for m in "${missing[@]}"; do
        echo "  - $m"
    done
    echo ""
    echo "Install them and re-run this script."
    exit 1
fi

echo "[1/5] Setting up ~/.memories/ config directory..."
mkdir -p ~/.memories

if [ -f ~/.memories/.env ]; then
    echo "  .env already exists, skipping."
else
    cp .env.example ~/.memories/.env

    read -rp "API port [8100]: " port
    port=${port:-8100}
    sed -i '' "s/API_PORT=8100/API_PORT=$port/" ~/.memories/.env 2>/dev/null || \
        sed -i "s/API_PORT=8100/API_PORT=$port/" ~/.memories/.env

    read -rp "API auth token (leave empty for no auth): " auth_token
    if [ -n "$auth_token" ]; then
        sed -i '' "s/API_AUTH_TOKEN=/API_AUTH_TOKEN=$auth_token/" ~/.memories/.env 2>/dev/null || \
            sed -i "s/API_AUTH_TOKEN=/API_AUTH_TOKEN=$auth_token/" ~/.memories/.env
    fi

    echo "  Created ~/.memories/.env"
fi

echo ""
echo "[2/5] Installing Python dependencies..."
uv sync

echo ""
echo "[3/5] Checking Ollama..."
if command -v ollama &>/dev/null; then
    echo "  Ollama found. Pulling nomic-embed-text..."
    ollama pull nomic-embed-text || echo "  Warning: could not pull model. Make sure Ollama is running."
else
    echo "  Ollama not found. Install it: brew install ollama"
    echo "  Then run: ollama pull nomic-embed-text"
fi

echo ""
echo "[4/5] Checking Docker..."
if docker info &>/dev/null; then
    echo "  Docker is running."
else
    echo "  Docker daemon not running. Start Docker Desktop first."
fi

echo ""
echo "[5/5] Setup complete!"
echo ""
echo "Quick start:"
echo "  make up      # Start Qdrant"
echo "  make serve   # Start API server"
echo "  uv run recall --help"
echo ""
echo "Or all at once:"
echo "  make dev"
