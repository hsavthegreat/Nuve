"""Tests for pipeline module."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import (
    extract_zip,
    replace_emojis_with_text,
    convert_bbox_x1y1x2y2_to_xywh,
    get_images_and_filenames,
)


def test_replace_emojis_with_text():
    """Test emoji replacement."""
    text = "I love this 😍"
    result = replace_emojis_with_text(text)
    assert "smile" in result.lower() or "heart" in result.lower()


def test_replace_emojis_none():
    """Test emoji replacement with None."""
    assert replace_emojis_with_text(None) is None


def test_convert_bbox():
    """Test bbox conversion."""
    x, y, w, h = convert_bbox_x1y1x2y2_to_xywh(10, 20, 50, 80)
    assert x == 10
    assert y == 20
    assert w == 40
    assert h == 60


def test_get_images_and_filenames_empty():
    """Test getting images from empty directory."""
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        images, filenames = get_images_and_filenames(tmpdir)
        assert len(images) == 0
        assert len(filenames) == 0


if __name__ == "__main__":
    test_replace_emojis_with_text()
    test_replace_emojis_none()
    test_convert_bbox()
    test_get_images_and_filenames_empty()
    print("All tests passed!")
