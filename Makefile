# TensorStream - Development Makefile
# Common development tasks and build commands

.PHONY: help install install-dev install-test install-docs install-all clean build test lint format docs serve-docs

# Default target
help:
	@echo "TensorStream Development Commands"
	@echo "================================="
	@echo ""
	@echo "Installation:"
	@echo "  install      Install production dependencies"
	@echo "  install-dev  Install development dependencies"
	@echo "  install-test Install testing dependencies"
	@echo "  install-docs Install documentation dependencies"
	@echo "  install-all  Install all dependencies"
	@echo ""
	@echo "Development:"
	@echo "  build        Build C++/CUDA extensions"
	@echo "  build-dev    Build in development mode"
	@echo "  clean        Clean build artifacts"
	@echo ""
	@echo "Code Quality:"
	@echo "  test         Run all tests"
	@echo "  test-unit    Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-gpu     Run GPU tests (requires CUDA)"
	@echo "  lint         Run all linting tools"
	@echo "  format       Format code with black and isort"
	@echo "  type-check   Run mypy type checking"
	@echo ""
	@echo "Documentation:"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"
	@echo "  docs-clean   Clean documentation build"
	@echo ""
	@echo "Package:"
	@echo "  dist         Build distribution packages"
	@echo "  upload-test  Upload to TestPyPI"
	@echo "  upload       Upload to PyPI"

# =============================================================================
# Installation Commands
# =============================================================================

install:
	pip install -r requirements-prod.txt

install-dev:
	pip install -r requirements-dev.txt

install-test:
	pip install -r requirements-test.txt

install-docs:
	pip install -r requirements-docs.txt

install-all:
	pip install -r requirements-all.txt

# =============================================================================
# Build Commands
# =============================================================================

build:
	python setup.py build_ext --inplace

build-dev:
	pip install -e .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name "*.so" -delete

# =============================================================================
# Testing Commands
# =============================================================================

test:
	pytest tests/ -v --cov=tensorstream --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v -m "not slow"

test-integration:
	pytest tests/integration/ -v

test-gpu:
	pytest tests/ -v -m "gpu"

test-fast:
	pytest tests/ -v -m "not slow and not gpu"

# =============================================================================
# Code Quality Commands
# =============================================================================

lint: lint-flake8 lint-mypy

lint-flake8:
	flake8 tensorstream/ tests/

lint-mypy:
	mypy tensorstream/

format:
	black tensorstream/ tests/
	isort tensorstream/ tests/

format-check:
	black --check tensorstream/ tests/
	isort --check-only tensorstream/ tests/

type-check:
	mypy tensorstream/

# =============================================================================
# Documentation Commands
# =============================================================================

docs:
	cd docs && make html

docs-clean:
	cd docs && make clean

serve-docs:
	cd docs/_build/html && python -m http.server 8000

docs-live:
	sphinx-autobuild docs docs/_build/html --host 0.0.0.0 --port 8000

# =============================================================================
# Package Distribution Commands
# =============================================================================

dist: clean
	python setup.py sdist bdist_wheel

upload-test: dist
	twine upload --repository testpypi dist/*

upload: dist
	twine upload dist/*

# =============================================================================
# Development Environment Commands
# =============================================================================

env-info:
	@echo "Python Version:"
	@python --version
	@echo ""
	@echo "PyTorch Version:"
	@python -c "import torch; print(torch.__version__)"
	@echo ""
	@echo "CUDA Available:"
	@python -c "import torch; print(torch.cuda.is_available())"
	@echo ""
	@echo "CUDA Version:"
	@python -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'N/A')"
	@echo ""
	@echo "GPU Count:"
	@python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"

benchmark:
	python scripts/benchmark.py

profile:
	python -m cProfile -s tottime scripts/profile_example.py

# =============================================================================
# Git Hooks
# =============================================================================

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files

# =============================================================================
# Docker Commands (if using Docker)
# =============================================================================

docker-build:
	docker build -t tensorstream:latest .

docker-test:
	docker run --rm tensorstream:latest make test

docker-run:
	docker run -it --rm tensorstream:latest bash
