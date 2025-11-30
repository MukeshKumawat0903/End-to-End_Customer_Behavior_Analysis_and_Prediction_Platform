# Makefile for common project tasks
# Usage: make <target>

.PHONY: help install test clean run-api run-dashboard format lint type-check all

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests with coverage"
	@echo "  make clean         - Clean temporary files"
	@echo "  make run-api       - Start API server"
	@echo "  make run-dashboard - Start Streamlit dashboard"
	@echo "  make format        - Format code with black"
	@echo "  make lint          - Lint code with flake8"
	@echo "  make type-check    - Type check with mypy"
	@echo "  make all           - Format, lint, type-check, and test"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	python run_tests.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete

run-api:
	python run_api.py

run-dashboard:
	streamlit run app/streamlit/dashboard.py

format:
	black src/ tests/ api.py run_api.py run_tests.py --line-length 100

lint:
	flake8 src/ tests/ api.py --max-line-length 100 --ignore E501,W503

type-check:
	mypy src/ --ignore-missing-imports

all: format lint type-check test
	@echo "✅ All checks passed!"
