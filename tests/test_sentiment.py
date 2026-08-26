"""Tests for sentiment analysis model."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sentiment.model import tokenize, clean_data, build_model


def test_tokenize():
    """Test tokenize function."""
    text = "I love this outfit! 😍"
    tokens = tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    # Should have converted emoji
    assert any("smile" in t or "heart" in t for t in tokens)


def test_tokenize_none():
    """Test tokenize with None input."""
    assert tokenize(None) is None


def test_clean_data():
    """Test clean_data function."""
    df = pd.DataFrame({
        'comments': ['great!', 'great!', 'okay'],
        'category': [1, 1, 0]
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 2  # duplicates removed
    assert list(cleaned['comments']) == ['great!', 'okay']


def test_build_model():
    """Test model building."""
    model = build_model()
    assert model is not None
    assert hasattr(model, 'fit')
    assert hasattr(model, 'predict')


if __name__ == "__main__":
    test_tokenize()
    test_tokenize_none()
    test_clean_data()
    test_build_model()
    print("All tests passed!")
