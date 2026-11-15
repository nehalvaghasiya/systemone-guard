.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= 3.11
EXTRAS ?= --extra dev --extra cpu
ARGS ?=
RUN = $(UV) run --locked --python $(PYTHON) $(EXTRAS)
CODE = src scripts spaces tests notebooks

.PHONY: help setup lock format format-check lint typecheck test test-artifacts coverage build verify-package audit check clean hooks data train evaluate benchmark demo prepare-space plots probes statistics smoke-extras

help: ## Show available development and research commands
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install the locked CPU development environment
	$(UV) sync --locked --python $(PYTHON) $(EXTRAS)

lock: ## Refresh the lockfile after changing dependencies (ARGS=--upgrade to upgrade)
	$(UV) lock $(ARGS)

format: ## Sort imports, wrap code, and space logical sections without code fixes
	$(RUN) ruff check --select I --fix $(CODE)
	$(RUN) ruff format $(CODE)
	$(RUN) python scripts/format_sections.py $(CODE)
	$(RUN) ruff format $(CODE)

format-check: ## Check imports, wrapping, and logical spacing without changing source
	$(RUN) ruff check --select I $(CODE)
	$(RUN) ruff format --check $(CODE)
	$(RUN) python scripts/format_sections.py --check $(CODE)

lint: ## Run Ruff lint checks
	$(RUN) ruff check $(CODE)

typecheck: ## Run strict mypy across all handwritten Python
	$(RUN) mypy

test: ## Run self-contained CPU unit and integration tests
	$(RUN) python -m pytest -q -m "not artifacts and not packaging and not optional" $(ARGS)

test-artifacts: ## Check existing local trained weights and datasets
	$(RUN) python -m pytest -q -m artifacts $(ARGS)

coverage: ## Report branch coverage for the self-contained suite
	$(RUN) python -m pytest -m "not artifacts and not packaging and not optional" --cov --cov-report=term-missing --cov-report=xml $(ARGS)

build: ## Build and validate wheel and source distribution
	$(UV) build --quiet --python $(PYTHON)
	$(RUN) twine check dist/systemone_guard-*.whl dist/systemone_guard-*.tar.gz

verify-package: build ## Install the wheel in an isolated environment and check inference
	JEV_UV="$(UV)" $(RUN) python -m pytest -q -m packaging $(ARGS)

audit: ## Audit the locked installed dependencies against vulnerability advisories
	$(RUN) python scripts/audit_dependencies.py $(ARGS)

check: format-check lint typecheck test ## Run all source-quality and test checks

clean: ## Remove generated caches and distributions; retain research assets
	rm -rf -- build dist .pytest_cache .ruff_cache .mypy_cache htmlcov src/*.egg-info
	rm -f -- .coverage .coverage.* coverage.xml
	find src scripts spaces tests -type d -name __pycache__ -prune -exec rm -rf -- {} +

hooks: ## Install optional local pre-commit hooks
	$(RUN) pre-commit install

data: ## Generate deterministic synthetic data (ARGS='--out ... --seed ...')
	$(RUN) python -m systemone_guard.data $(ARGS)

train: ## Train the guard (ARGS='--data ... --out ...')
	$(RUN) python -m systemone_guard.train $(ARGS)

evaluate: ## Evaluate saved models (ARGS='--split test --out ...')
	$(RUN) python -m systemone_guard.evaluate $(ARGS)

benchmark: ## Run the opt-in pretrained benchmark; may download model weights
	$(RUN) --extra llm python -m systemone_guard.benchmarking.pretrained $(ARGS)

demo: ## Launch the Gradio demo using an existing checkpoint
	$(RUN) --extra demo python spaces/app.py $(ARGS)

prepare-space: ## Prepare a standalone local HF Space export
	$(RUN) python scripts/huggingface.py prepare $(ARGS)

plots: ## Plot existing evaluation reports
	$(RUN) python scripts/plots.py results $(ARGS)

probes: ## Evaluate the manually authored diagnostic probes
	$(RUN) python scripts/evaluate_probes.py $(ARGS)

statistics: ## Bootstrap existing evaluation predictions
	$(RUN) python scripts/bootstrap_summary.py $(ARGS)

smoke-extras: ## Validate optional adapters without model downloads or publication
	$(RUN) --extra demo --extra llm --extra publish python -m pytest -q -m optional $(ARGS)
