PYTHON ?= .venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help install run docker-run corpus test lint clean

help: ## Show the available commands
	@grep -E "^[a-z-]+:.*## " $(MAKEFILE_LIST) | awk -F ":.*## " '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Install the Python dependencies
	test -d .venv || python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

run: ## Run the API locally (needs a running Neo4j)
	$(PYTHON) -m uvicorn app.main:app --reload

docker-run: ## Build and start Neo4j and the API with Docker Compose
	docker compose up --build

corpus: ## Download the Wikipedia corpus into data/raw
	$(PYTHON) -m scripts.download_corpus

test: ## Run the tests with coverage (fails under 60%)
	$(PYTHON) -m pytest --cov=app --cov=scripts --cov-report=term-missing --cov-fail-under=60

lint: ## Run pylint (fails under 9.5)
	$(PYTHON) -m pylint app scripts tests --fail-under=9.5

clean: ## Remove caches and coverage files
	rm -rf .pytest_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
