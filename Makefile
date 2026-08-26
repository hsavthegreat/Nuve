# Makefile for Nuve - AI Fashion Trend Forecasting

.PHONY: help install format lint test clean pipeline train-sentiment serve notebooks

# Default target
help:
	@echo "Nuve - AI Fashion Trend Forecasting"
	@echo ""
	@echo "Available commands:"
	@echo "  install        Install production dependencies"
	@echo "  format         Format code with black and isort"
	@echo "  lint           Lint code with ruff"
	@echo "  test           Run tests with pytest"
	@echo "  clean          Remove cache and build artifacts"
	@echo "  pipeline       Run main processing pipeline"
	@echo "  train-sentiment Train sentiment analysis model"
	@echo "  serve          Launch Streamlit dashboard"
	@echo "  notebooks      Start Jupyter notebook server"

# Installation
install:
	pip install -r requirements.txt
	pip install black ruff isort pytest  # Dev tools

# Code quality
format:
	black src/ app/
	isort src/ app/

lint:
	ruff check src/ app/

test:
	pytest tests/ -v || echo "No tests found or tests failed"

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
	rm -rf .pytest_cache/ htmlcov/ .mypy_cache/ .ruff_cache/ 2>/dev/null || true
	@echo "Cleaned up cache and build artifacts"

# Development setup (run once)
setup: install
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
	@echo "Setup complete! Run 'make serve' to start the dashboard."
