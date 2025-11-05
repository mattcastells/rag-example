COMPOSE=docker compose -f docker-compose.yml

.PHONY: up down logs migrate ingest-local test fmt lint

up:
$(COMPOSE) up -d --build

down:
$(COMPOSE) down -v

logs:
$(COMPOSE) logs -f backend

migrate:
$(COMPOSE) run --rm backend alembic upgrade head

ingest-local:
$(COMPOSE) run --rm backend python -m app.ingest.pipeline --path ./data/docs --repo local --tag local --acl public

test:
$(COMPOSE) run --rm backend pytest -q

fmt:
$(COMPOSE) run --rm backend bash -c "ruff check --fix . && black ."

lint:
$(COMPOSE) run --rm backend bash -c "ruff check ."
