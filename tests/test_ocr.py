"""
test_ocr.py — Smoke tests for OCR preprocessing (no Tesseract needed).

Run with:
    python3 -m pytest tests/test_ocr.py -v

These tests only exercise the image preprocessing and blur detection
functions.  They do NOT call Tesseract, so they work on any machine
with numpy and opencv installed.

This is useful for checking that the preprocessing pipeline doesn't
crash on edge cases (empty images, greyscale input, etc.).
"""

import numpy as np
import pytest

from src.ocr import PlateOCR


@pytest.fixture
def ocr():
    """Create a PlateOCR instance with default config for testing.

    In pytest, a 'fixture' is like setUp() in PHPUnit — it runs before
    each test method and provides the test with ready-to-use objects.
    """
    cfg = {
        "ocr": {
            "confidence_threshold": 50,
            "max_blur_variance": 100.0,
            "tesseract_config": "--psm 7 --oem 3",
            "char_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        }
    }
    return PlateOCR(cfg)


class TestBlur:
    """Tests for blur_score() — Laplacian variance sharpness metric."""

    def test_sharp_image(self, ocr):
        # Random noise has lots of edges → high variance → "sharp"
        img = np.random.randint(0, 255, (60, 200), dtype=np.uint8)
        score = ocr.blur_score(img)
        assert score > 100, f"Expected sharp (>100), got {score}"

    def test_flat_image(self, ocr):
        # A solid grey image has zero edges → variance ≈ 0 → "blurry"
        img = np.full((60, 200), 128, dtype=np.uint8)
        score = ocr.blur_score(img)
        assert score < 1, f"Expected blurry (<1), got {score}"


class TestPreprocess:
    """Tests for preprocess() — the image cleaning pipeline."""

    def test_output_shape(self, ocr):
        # Input: 30px tall colour image
        plate = np.random.randint(0, 255, (30, 120, 3), dtype=np.uint8)
        out = ocr.preprocess(plate)
        # Output should be resized to 60px + 10px border top + 10px bottom = 80px
        assert out.shape[0] == 80

    def test_grayscale_input(self, ocr):
        # Should handle greyscale input without crashing
        plate = np.random.randint(0, 255, (40, 160), dtype=np.uint8)
        out = ocr.preprocess(plate)
        # Output should still be single-channel (greyscale)
        assert len(out.shape) == 2

    def test_empty_returns_as_is(self, ocr):
        # Empty array should pass through without error
        empty = np.array([], dtype=np.uint8)
        assert ocr.preprocess(empty).size == 0
