# Makefile for Perspective World Model Kit (PWMK)
# Provides convenient development and testing commands

.PHONY: help install install-dev test test-fast test-slow test-gpu lint format type-check security docs docs-serve clean build publish benchmark integration coverage

# Default target
help:
	@echo "PWMK Development Commands"
	@echo "========================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  install         Install package in development mode"
	@echo "  install-dev     Install with all development dependencies"
	@echo "  install-all     Install with all optional dependencies"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test            Run all tests with coverage"
	@echo "  test-fast       Run only fast tests (unit tests)"
	@echo "  test-slow       Run slow tests (integration, e2e)"
	@echo "  test-gpu        Run GPU-accelerated tests"
	@echo "  test-research   Run research validation tests"
	@echo "  benchmark       Run performance benchmarks"
	@echo "  integration     Run integration tests with services"
	@echo "  coverage        Generate detailed coverage report"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint            Run all linting tools"
	@echo "  format          Auto-format code with black and isort"
	@echo "  type-check      Run mypy type checking"
	@echo "  security        Run security scanners"
	@echo "  pre-commit      Run pre-commit hooks on all files"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs            Build documentation"
	@echo "  docs-serve      Serve documentation locally"
	@echo "  docs-clean      Clean documentation build"
	@echo ""
	@echo "Distribution Commands:"
	@echo "  build           Build package distributions"
	@echo "  publish         Publish to PyPI (requires credentials)"
	@echo "  publish-test    Publish to Test PyPI"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  clean           Clean build artifacts and caches"
	@echo "  clean-all       Deep clean including virtual environments"
	@echo "  update-deps     Update development dependencies"

# Setup commands
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test]"

install-all:
	pip install -e ".[dev,test,unity,prolog,docs]"

# Testing commands
test:
	pytest tests/ -v --cov=pwmk --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -m "not slow and not gpu and not integration" --maxfail=1

test-slow:
	pytest tests/ -v -m "slow or integration" --tb=short

test-gpu:
	pytest tests/ -v -m gpu --tb=short

test-research:
	pytest tests/ -v -m research --tb=short

test-parallel:
	pytest tests/ -v -n auto --cov=pwmk --cov-report=term-missing

benchmark:
	pytest tests/benchmarks/ -v --benchmark-only --benchmark-sort=mean --benchmark-json=benchmark.json

integration:
	docker-compose -f docker-compose.yml up -d
	pytest tests/integration/ -v --tb=short
	docker-compose -f docker-compose.yml down

coverage:
	coverage erase
	coverage run -m pytest tests/
	coverage report --show-missing
	coverage html
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality commands
lint:
	black --check --diff pwmk tests
	isort --check-only --diff pwmk tests
	flake8 pwmk tests
	ruff check pwmk tests

format:
	black pwmk tests
	isort pwmk tests
	@echo "Code formatted successfully"

type-check:
	mypy pwmk --ignore-missing-imports

security:
	bandit -r pwmk -ll
	safety check --json --output safety-report.json
	semgrep --config=auto pwmk

pre-commit:
	pre-commit run --all-files

# Documentation commands
docs:
	cd docs && make html
	@echo "Documentation built in docs/_build/html/index.html"

docs-serve:
	cd docs && make livehtml
	@echo "Documentation server running at http://localhost:8000"

docs-clean:
	cd docs && make clean

# Distribution commands  
build:
	python -m build
	@echo "Package built in dist/"

publish: build
	twine check dist/*
	twine upload dist/*

publish-test: build
	twine check dist/*
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Maintenance commands
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/
	rm -f coverage.xml benchmark.json safety-report.json

clean-all: clean
	rm -rf .tox/ .venv/ venv/ env/
	docker system prune -f

update-deps:
	pip-compile requirements/dev.in
	pip-compile requirements/test.in
	pip-compile requirements/docs.in
	pre-commit autoupdate

# Environment setup
setup-env:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev,test]"
	.venv/bin/pre-commit install
	@echo "Development environment setup complete"
	@echo "Activate with: source .venv/bin/activate"

# Tox commands for comprehensive testing
test-all:
	tox

test-py39:
	tox -e py39

test-py310:
	tox -e py310

test-py311:
	tox -e py311

test-py312:
	tox -e py312

# Research and experimentation
jupyter:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Unity development (if Unity environments are used)
setup-unity:
	./scripts/setup_unity.sh

build-unity:
	cd unity && ./build.sh

# Database and monitoring (if needed)
setup-monitoring:
	docker-compose -f monitoring/docker-compose.yml up -d
	@echo "Monitoring stack available at:"
	@echo "  Grafana: http://localhost:3000"
	@echo "  Prometheus: http://localhost:9090"

# Development workflow shortcuts
dev-setup: install-dev pre-commit setup-monitoring
	@echo "Development environment ready!"

dev-test: format lint type-check test-fast
	@echo "Quick development checks passed!"

ci-test: lint type-check security test coverage
	@echo "CI validation complete!"

# Release workflow
release-check: clean lint type-check test-all security docs
	@echo "Release checks passed - ready for publishing!"

# Utility commands
show-coverage:
	@echo "Opening coverage report..."
	python -c "import webbrowser; webbrowser.open('htmlcov/index.html')"

show-docs:
	@echo "Opening documentation..."
	python -c "import webbrowser; webbrowser.open('docs/_build/html/index.html')"

# Variables for customization
PYTHON ?= python
PIP ?= pip
PYTEST_ARGS ?= -v
COVERAGE_THRESHOLD ?= 80