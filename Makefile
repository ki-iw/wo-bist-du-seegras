.PHONY: help install format check test build clean-build docs docs-test

install: ## Install the project dependencies, pre-commit hooks and Jupyter notebook filters.
	@echo "🏗️ Installing project dependencies"
	@poetry install
	@echo "🪐 Installing Jupyter cleaner"
	@git config --local filter.jupyter.smudge cat
	@echo "🪝 Installing pre-commit hooks"
	@poetry run pre-commit install
	@echo "🎉 Done"

format: ## Format all project files and sort imports.
	@echo "🪄 Formatting files"
	@poetry run black .
	@poetry run ruff --select I --fix .

check: ## Run code quality checks. Recommended before committing.
	@echo "🔎 Checking Poetry lock file consistency with pyproject.toml"
	@poetry lock --check
	@echo "🔍 Running pre-commit"
	@poetry run pre-commit run -a



test: ## Test the code with pytest
	@echo "🚦 Testing code"
	@poetry run pytest --doctest-modules

dev: format check test

build: clean-build ## Build wheel file using poetry
	@echo "🚀 Creating wheel file"
	@poetry build

clean-build: ## clean build artifacts
	@rm -rf dist


help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help