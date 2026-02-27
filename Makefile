.PHONY: up down serve dev test test-integration test-degraded test-all status install

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
	uv run pytest tests/test_unit.py -v

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
