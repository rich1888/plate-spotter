"""
test_uk_plate.py — Unit tests for UK plate validation and dateless detection.

Run with:
    python3 -m pytest tests/test_uk_plate.py -v

These tests don't need any hardware, camera, or Tesseract installed.
They test the pure-logic functions that decide whether an OCR result
looks like a valid UK plate.

pytest basics (for PHP developers):
  - Each class groups related tests (like a PHPUnit test class).
  - Methods starting with test_ are automatically discovered and run.
  - @pytest.mark.parametrize runs the same test with different inputs
    (like a PHPUnit @dataProvider).
  - assert is like PHPUnit's $this->assertEquals() — it checks a condition.
"""

import pytest

from src.uk_plate import (
    normalize_plate,
    is_plausible_uk_plate,
    is_dateless,
    classify_plate,
    matches_known_format,
)


# ── Normalisation tests ──────────────────────────────────────────────── #

class TestNormalize:
    """Tests for normalize_plate() — cleaning raw OCR output."""

    def test_basic(self):
        # Spaces should be removed
        assert normalize_plate("AB12 CDE") == "AB12CDE"

    def test_hyphens_spaces(self):
        # Hyphens and spaces stripped, lowercase → uppercase
        assert normalize_plate("ab-12-cde") == "AB12CDE"

    def test_lowercase(self):
        assert normalize_plate("ab12cde") == "AB12CDE"

    def test_junk(self):
        # All non-alphanumeric characters stripped
        assert normalize_plate("A.B!1@2#C$D%E") == "AB12CDE"

    def test_empty(self):
        assert normalize_plate("") == ""


# ── Format matching tests ────────────────────────────────────────────── #

class TestFormatMatch:
    """Tests for matches_known_format() — identifying the plate style."""

    @pytest.mark.parametrize("plate,expected", [
        # Current format (2001+): 2 letters + 2 digits + 3 letters
        ("AB12CDE", "current"),
        ("LB68VFR", "current"),

        # Prefix format (1983–2001): 1 letter + 1–3 digits + 3 letters
        ("A123ABC", "prefix"),
        ("A1ABC",   "prefix"),

        # Suffix format (1963–1983): 3 letters + 1–3 digits + 1 letter
        ("ABC123A", "suffix"),
        ("ABC1A",   "suffix"),

        # Dateless — digits first
        ("1234ABC", "dateless_num_alpha"),
        ("1A",      "dateless_num_alpha"),

        # Dateless — letters first
        ("ABC1234", "dateless_alpha_num"),
        ("A1",      "dateless_alpha_num"),
    ])
    def test_known_formats(self, plate, expected):
        assert matches_known_format(plate) == expected

    def test_no_match_garbage(self):
        # Pure letters with no digits shouldn't match anything
        assert matches_known_format("ZZZZZZZZ") is None

    def test_confusion_swap(self):
        # "AB1OCDE" has a letter O where a digit 0 should be.
        # The confusion-swap logic should still recognise it.
        assert matches_known_format("AB1OCDE") is not None


# ── Plausibility tests ───────────────────────────────────────────────── #

class TestPlausibility:
    """Tests for is_plausible_uk_plate() — the main filter."""

    @pytest.mark.parametrize("plate", [
        "AB12CDE",   # current format
        "A123ABC",   # prefix format
        "ABC123A",   # suffix format
        "1234ABC",   # dateless (digits first)
        "ABC1234",   # dateless (letters first)
        "1A",        # shortest dateless
        "A1",        # shortest dateless
    ])
    def test_valid_plates(self, plate):
        assert is_plausible_uk_plate(plate) is True

    @pytest.mark.parametrize("plate", [
        "",           # empty string
        "A",          # too short (min_length=2)
        "ABCDEFGHI",  # too long (9 chars, max=8)
        "AAAA",       # all letters, no digits
        "1111",       # all digits, no letters
        "AAAAAAA",    # all same character
    ])
    def test_invalid_plates(self, plate):
        assert is_plausible_uk_plate(plate) is False

    def test_all_same_rejected(self):
        assert is_plausible_uk_plate("AAAA", reject_all_same=True) is False

    def test_all_same_allowed(self):
        # Even with reject_all_same=False, "AAAA" still fails
        # because it has no digits
        assert is_plausible_uk_plate("AAAA", reject_all_same=False) is False

    def test_configurable_length(self):
        # "A1" is 2 chars — should fail with min_length=3
        assert is_plausible_uk_plate("A1", min_length=3) is False
        # …but pass with min_length=2
        assert is_plausible_uk_plate("A1", min_length=2) is True


# ── Dateless detection tests ─────────────────────────────────────────── #

class TestDateless:
    """Tests for is_dateless() — the user's dateless rule.

    Rule: a plate is dateless if it starts OR ends with a digit.
    """

    @pytest.mark.parametrize("plate,expected", [
        ("AB12CDE", False),   # starts A, ends E → NOT dateless
        ("A123ABC", False),   # starts A, ends C → NOT dateless
        ("ABC123A", False),   # starts A, ends A → NOT dateless
        ("1234ABC", True),    # starts with 1 → dateless
        ("ABC1234", True),    # ends with 4 → dateless
        ("1A",      True),    # starts with 1 → dateless
        ("A1",      True),    # ends with 1 → dateless
        ("123",     True),    # starts AND ends with digit → dateless
    ])
    def test_dateless(self, plate, expected):
        assert is_dateless(plate) is expected

    def test_empty(self):
        assert is_dateless("") is False


# ── Full classification tests ────────────────────────────────────────── #

class TestClassify:
    """Tests for classify_plate() — the complete pipeline."""

    def test_current_format(self):
        result = classify_plate("AB12 CDE")
        assert result is not None
        assert result["normalized"] == "AB12CDE"
        assert result["format"] == "current"
        assert result["dateless"] is False

    def test_dateless_digits_first(self):
        result = classify_plate("1234 ABC")
        assert result is not None
        assert result["dateless"] is True

    def test_garbage_returns_none(self):
        # Nonsense input should return None
        assert classify_plate("@#$%^") is None

    def test_raw_preserved(self):
        # The original input should be kept in the "raw" field
        result = classify_plate("ab 12 cde")
        assert result is not None
        assert result["raw"] == "ab 12 cde"
        assert result["normalized"] == "AB12CDE"
