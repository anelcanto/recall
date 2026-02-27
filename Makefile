.PHONY: up down serve dev test test-integration test-degraded test-all lint status install hooks _release-preflight release-patch release-minor release-major

up:
	docker run -d --name qdrant \
		-p 127.0.0.1:6333:6333 \
		-p 127.0.0.1:6334:6334 \
		-v qdrant_data:/qdrant/storage \
		qdrant/qdrant:v1.13.2 || docker start qdrant

down:
	docker stop qdrant && docker rm qdrant

serve:
	uv run uvicorn memory_api.main:app --host 127.0.0.1 --port 8100

dev:
	$(MAKE) up && $(MAKE) serve

test:
	uv run pytest tests/test_unit.py tests/test_cli.py tests/test_mcp.py tests/test_embeddings.py -v

lint:
	uv run ruff check .
	uv run ruff format --check .

hooks:
	uv run pre-commit install

test-integration:
	uv run pytest tests/test_integration.py -v

test-degraded:
	uv run pytest tests/test_degraded.py -v

test-all:
	uv run pytest tests/ -v

status:
	@curl -sf http://127.0.0.1:8100/health | python3 -m json.tool || echo "API not running"

install:
	@mkdir -p ~/.recall
	@test -f ~/.recall/.env || cp .env.example ~/.recall/.env
	uv sync
	@echo ""
	@echo "Installed. Run 'make dev' to start Qdrant + API server."
	@echo "CLI available via: uv run recall --help"

# Release targets — bumps version, commits, tags, then push to trigger CI → PyPI
_release-preflight:
	@if [ "$$(git branch --show-current)" != "main" ]; then \
		echo "Error: must be on main branch to release"; exit 1; fi
	@if ! git diff --quiet || ! git diff --cached --quiet; then \
		echo "Error: working tree is not clean"; exit 1; fi
	@git fetch origin main --quiet
	@if [ "$$(git rev-parse main)" != "$$(git rev-parse origin/main)" ]; then \
		echo "Error: local main is behind origin — run git pull first"; exit 1; fi

release-patch: _release-preflight
	uv run bump-my-version bump patch
	git push origin main $$(git describe --tags --abbrev=0)

release-minor: _release-preflight
	uv run bump-my-version bump minor
	git push origin main $$(git describe --tags --abbrev=0)

release-major: _release-preflight
	uv run bump-my-version bump major
	git push origin main $$(git describe --tags --abbrev=0)
