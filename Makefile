.PHONY: help install format check test build clean-build docs docs-test

install:
	@echo "🏗️ Installing project dependencies"
	@poetry install
	@echo "🪐 Installing Jupyter cleaner"
	@git config --local filter.jupyter.smudge cat
	@echo "🪝 Installing pre-commit hooks"
	@poetry run pre-commit install
	@echo "Installing FiftyOne"
	@pip install fiftyone
	@echo "🎉 Done"

format:
	@echo "🪄 Formatting files"
	@poetry run black .
	@poetry run ruff --select I --fix .

check:
	@echo "🔎 Checking Poetry lock file consistency with pyproject.toml"
	@poetry check
	@echo "🔍 Running pre-commit"
	@poetry run pre-commit run -a

test:
	@echo "🚦 Testing code"
	@poetry run pytest -vv -p no:warnings --cov=zug_seegras/core  tests

build: clean-build
	@echo "🚀 Creating wheel file"
	@poetry build

dev: format check test

clean-build:
	@rm -rf dist

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help