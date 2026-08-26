# Makefile for Nuve - AI Fashion Trend Forecasting

.PHONY: help install install-dev format lint type-check test clean pipeline train-sentiment serve notebooks

# Default target
help:
	@echo "Nuve - AI Fashion Trend Forecasting"
	@echo ""
	@echo "Available commands:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  format         Format code with black and isort"
	@echo "  lint           Lint code with ruff"
	@echo "  type-check     Type check with mypy"
	@echo "  test           Run tests with pytest"
	@echo "  clean          Remove cache and build artifacts"
	@echo "  pipeline       Run main processing pipeline"
	@echo "  train-sentiment Train sentiment analysis model"
	@echo "  serve          Launch Streamlit dashboard"
	@echo "  notebooks      Start Jupyter notebook server"
	@echo "  pre-commit     Install pre-commit hooks"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# Code quality
format:
	black src/ app/ tests/
	isort src/ app/ tests/

lint:
	ruff check src/ app/ tests/

type-check:
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Combined quality check
check: format lint type-check test

# Pre-commit
pre-commit:
	pre-commit run --all-files

# Main pipeline
pipeline:
	python -m src.pipeline

# Train sentiment model
train-sentiment:
	python -m src.sentiment.model

# Serve dashboard
serve:
	streamlit run app/streamlit_app.py

# Notebooks
notebooks:
	jupyter notebook notebooks/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ htmlcov/ .mypy_cache/ .ruff_cache/ dist/ build/ *.egg-info/ 2>/dev/null || true
	@echo "Cleaned up cache and build artifacts"

# Development setup (run once)
setup: install-dev
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
	@echo "Setup complete! Run 'make serve' to start the dashboard."

# Install system dependencies (Ubuntu/Debian)
install-system:
	sudo apt-get update && sudo apt-get install -y \
		python3-venv python3-dev \
		libgl1-mesa-glx libglib2.0-0 \
		ffmpeg libsm6 libxext6 \
		git

# Full CI simulation locally
ci: clean install-dev check
	@echo "CI simulation complete!"
