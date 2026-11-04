# JobMatch — developer commands. Run `make` (or `make help`) to list targets.
SHELL := /bin/bash
.DEFAULT_GOAL := help

COMPOSE ?= docker compose
# Every service except the api (useful while running uvicorn on the host).
INFRA_SERVICES := app-db prometheus grafana langfuse-web langfuse-worker \
                  langfuse-postgres clickhouse redis minio

.PHONY: help up down logs ps infra test lint format eval qwen clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up: ## Start the full stack (builds the api image)
	$(COMPOSE) up -d --build

down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

logs: ## Follow logs (make logs S=api for a single service)
	$(COMPOSE) logs -f --tail=100 $(S)

ps: ## Show service status
	$(COMPOSE) ps

infra: ## Start everything except the api
	$(COMPOSE) up -d $(INFRA_SERVICES)

test: ## Unit tests (no LLM needed)
	uv run pytest -m "not llm"

lint: ## Ruff lint + format check
	uv run ruff check . && uv run ruff format --check .

format: ## Auto-fix lint issues and format code
	uv run ruff check --fix . && uv run ruff format .

eval: ## DeepEval suite (needs llama-server; ARGS="--cases a,b --refresh --no-langfuse")
	uv run --group eval python -m evals.run_eval $(ARGS)

qwen: ## Print the command starting the host LLM (llama-server)
	@echo "Run on the host (not in Docker):  ~/start_qwen_light.sh"
	@echo "Then check:  curl -s http://172.17.0.1:8080/v1/models"

clean: ## Stop the stack AND delete all volumes (asks for confirmation)
	@read -r -p "This deletes ALL data volumes (Langfuse, app-db, Grafana, Prometheus). Continue? [y/N] " ans; \
		if [[ "$$ans" == [yY] ]]; then $(COMPOSE) down -v; else echo "Aborted."; fi
