.PHONY: help install install-dev setup lint format type-check test test-watch test-cov clean clean-logs clean-all build backup-db install-spacy dev-setup ci logs logs-follow logs-errors logs-discord logs-clean debug

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation and setup
install: ## Install production dependencies
	poetry install --only main

install-dev: ## Install all dependencies including dev dependencies
	poetry install

install-spacy: ## Install spaCy English model
	poetry run python scripts/install_spacy.py

setup-db: ## Initialize the database
	poetry run python scripts/setup_database.py

dev-setup: install-dev install-spacy setup-db ## Complete development environment setup
	@echo "🎉 Development environment is ready!"

# Code quality and linting (this replaces your 4 poetry commands!)
format: ## Format code with black and isort
	poetry run black src/ tests/
	poetry run isort src/ tests/

lint: ## Run all linting checks (black, isort, flake8, mypy)
	@echo "🔍 Running code formatting checks..."
	poetry run black --check src/ tests/
	poetry run isort --check-only src/ tests/
	@echo "🔍 Running flake8..."
	poetry run flake8 src/ tests/
	@echo "🔍 Running mypy..."
	poetry run mypy src/

lint-fix: format ## Fix all auto-fixable linting issues
	@echo "✨ Code formatted successfully!"

type-check: ## Run only type checking
	poetry run mypy src/

# Testing
test: ## Run all tests
	poetry run pytest

test-unit: ## Run only unit tests
	poetry run pytest -m "unit"

test-integration: ## Run only integration tests
	poetry run pytest -m "integration"

test-watch: ## Run tests in watch mode
	poetry run pytest-watch

test-cov: ## Run tests with coverage report
	poetry run pytest --cov-report=html --cov-report=term

test-fast: ## Run tests without coverage
	poetry run pytest --no-cov -x

# Database operations
backup-db: ## Backup the database
	poetry run python scripts/backup_db.py

# Development utilities
clean: ## Clean up temporary files and caches
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-logs: ## Clean all log files
	@echo "🧹 Cleaning all log files..."
	@rm -rf logs/*.log logs/*.log.* 2>/dev/null || true
	@echo "✅ All log files cleaned!"

clean-all: clean clean-logs ## Clean everything including logs
	@echo "🧹 Complete cleanup finished!"

build: ## Build the package
	poetry build

run: ## Run the pepino CLI
	poetry run pepino

# CI/CD pipeline simulation
ci: lint test ## Run CI pipeline locally (lint + test)
	@echo "✅ CI pipeline completed successfully!"

# Quick development cycle
dev: lint-fix test-fast ## Quick development cycle: format code and run fast tests
	@echo "🚀 Ready for development!"

# Pre-commit hook simulation
pre-commit: lint test-unit ## Run pre-commit checks
	@echo "✅ Pre-commit checks passed!"

# Logging and debugging
logs: ## View recent application logs
	@echo "📋 Recent application logs:"
	@if [ -f logs/pepino.log ]; then tail -50 logs/pepino.log; else echo "No logs found. Run the application first."; fi

logs-follow: ## Follow application logs in real-time
	@echo "📋 Following application logs (Ctrl+C to stop):"
	@tail -f logs/pepino.log 2>/dev/null || echo "No logs found. Run the application first."

logs-errors: ## View recent error logs
	@echo "🔴 Recent error logs:"
	@if [ -f logs/errors.log ]; then tail -50 logs/errors.log; else echo "No error logs found."; fi

logs-discord: ## View Discord bot logs
	@echo "🤖 Recent Discord bot logs:"
	@if [ -f logs/discord.log ]; then tail -50 logs/discord.log; else echo "No Discord logs found."; fi

logs-clean: ## Clean old log files
	@echo "🧹 Cleaning old log files..."
	@find logs/ -name "*.log.*" -mtime +7 -delete 2>/dev/null || true
	@echo "✅ Old log files cleaned!"

debug: ## Run in debug mode with verbose logging
	poetry run pepino --verbose 