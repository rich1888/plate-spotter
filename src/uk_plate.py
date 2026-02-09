"""
uk_plate.py — UK number plate validation and dateless detection.

This module answers three questions:
  1. "Is this string a plausible UK registration?"  → is_plausible_uk_plate()
  2. "Is this a dateless plate?"                    → is_dateless()
  3. "Give me everything about this plate"          → classify_plate()

It does NOT query any external database — it uses regex patterns and
heuristics to decide whether an OCR result *looks like* a real UK plate.

UK plate formats recognised
────────────────────────────
  Current (2001+):  AB12 CDE        e.g. LB68 VFR
  Prefix (1983–01): A123 ABC        e.g. P456 XYZ
  Suffix (1963–83): ABC 123A        e.g. KLM 321D
  Dateless:         1–4 digits + 1–3 letters   OR
                    1–3 letters + 1–4 digits
  Diplomatic:       123D456  or  123X456

Dateless rule (as defined by user)
──────────────────────────────────
  A plate is "dateless" when its normalised string **starts OR ends
  with a digit**.
"""

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Character-confusion map
#
# OCR engines commonly swap visually similar characters.  This map lets us
# try a single swap and recheck against known formats.
# We keep it conservative — only single-character replacements.
# ---------------------------------------------------------------------------
CHAR_CONFUSIONS = {
    "O": "0", "0": "O",   # letter O  ↔  digit zero
    "I": "1", "1": "I",   # letter I  ↔  digit one
    "S": "5", "5": "S",   # letter S  ↔  digit five
    "Z": "2", "2": "Z",   # letter Z  ↔  digit two
    "B": "8", "8": "B",   # letter B  ↔  digit eight
    "G": "6", "6": "G",   # letter G  ↔  digit six
    "D": "0",              # letter D  →  digit zero (one-way)
    "Q": "0",              # letter Q  →  digit zero (one-way)
}

# ---------------------------------------------------------------------------
# Regex patterns for known UK plate formats
#
# Each pattern operates on NORMALISED text: upper-case, no spaces/hyphens.
# re.compile() pre-compiles them for speed — the regex is only parsed once.
# ---------------------------------------------------------------------------
UK_FORMATS = {
    # Current (2001+): two letters, two digits, three letters
    "current": re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$"),

    # Prefix (1983–2001): one letter, 1–3 digits, three letters
    "prefix": re.compile(r"^[A-Z][0-9]{1,3}[A-Z]{3}$"),

    # Suffix (1963–1983): three letters, 1–3 digits, one letter
    "suffix": re.compile(r"^[A-Z]{3}[0-9]{1,3}[A-Z]$"),

    # Dateless — digits first: 1–4 digits then 1–3 letters  (e.g. 1234 ABC)
    "dateless_num_alpha": re.compile(r"^[0-9]{1,4}[A-Z]{1,3}$"),

    # Dateless — letters first: 1–3 letters then 1–4 digits  (e.g. ABC 1234)
    "dateless_alpha_num": re.compile(r"^[A-Z]{1,3}[0-9]{1,4}$"),

    # Diplomatic: three digits, D or X, three digits  (e.g. 123D456)
    "diplomatic": re.compile(r"^[0-9]{3}[DX][0-9]{3}$"),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Public functions
# ═══════════════════════════════════════════════════════════════════════════

def normalize_plate(raw_text: str) -> str:
    """Convert raw OCR output to a clean plate string.

    - Converts to upper case
    - Strips everything except A-Z and 0-9

    Examples:
        "ab12 cde"  →  "AB12CDE"
        "A.B!1@2"   →  "AB12"
    """
    return re.sub(r"[^A-Z0-9]", "", raw_text.upper())


def matches_known_format(text: str) -> Optional[str]:
    """Check *text* against known UK plate regex patterns.

    Tries the exact text first, then single-character confusion-swap
    variants (e.g. O↔0) so that a minor OCR error doesn't discard an
    otherwise valid plate.

    Args:
        text: Normalised plate string (upper-case, no spaces).

    Returns:
        The format name (e.g. "current", "prefix") or None.
    """
    # --- Pass 1: check the original text against every format ---
    for name, rx in UK_FORMATS.items():
        if rx.match(text):
            return name

    # --- Pass 2: try single-character swaps, then recheck ---
    for variant in _single_swap_variants(text):
        if variant == text:
            continue  # already checked above
        for name, rx in UK_FORMATS.items():
            if rx.match(variant):
                return name

    return None


def is_plausible_uk_plate(
    text: str,
    min_length: int = 2,
    max_length: int = 8,
    reject_all_same: bool = True,
) -> bool:
    """Heuristic check: could *text* plausibly be a UK registration?

    This is deliberately permissive — it's a pre-filter to stop obvious
    garbage reaching the API, not a definitive lookup.

    Args:
        text:            Normalised plate string.
        min_length:      Shortest string we'll accept.
        max_length:      Longest string we'll accept.
        reject_all_same: Discard strings like "AAAA" or "1111".

    Returns:
        True if the plate looks plausible.
    """
    # Basic sanity checks
    if not text or not text.isalnum():
        return False
    if len(text) < min_length or len(text) > max_length:
        return False
    if reject_all_same and len(set(text)) == 1:
        return False

    # A valid UK plate always has at least one letter AND one digit
    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)
    if not (has_letter and has_digit):
        return False

    # If it matches a known format (or a confusion variant does), accept it
    if matches_known_format(text):
        return True

    # Fallback heuristic for unusual plates that don't match known formats:
    # reject wildly unbalanced letter/digit ratios (e.g. 7 letters + 1 digit)
    letters = sum(c.isalpha() for c in text)
    digits = len(text) - letters
    if len(text) >= 5 and (letters == 0 or digits == 0):
        return False
    ratio = max(letters, digits) / max(min(letters, digits), 1)
    if ratio > 6:
        return False

    return True


def is_dateless(text: str) -> bool:
    """Check whether a plate is "dateless".

    Rule (as specified):
        A plate is dateless when its normalised string **starts with a
        digit** OR **ends with a digit**.

    Args:
        text: Normalised plate string.

    Returns:
        True if dateless.

    Examples:
        "AB12CDE"  → False  (starts A, ends E)
        "1234ABC"  → True   (starts with 1)
        "ABC1234"  → True   (ends with 4)
    """
    if not text:
        return False
    return text[0].isdigit() or text[-1].isdigit()


def classify_plate(
    raw_text: str,
    min_length: int = 2,
    max_length: int = 8,
    reject_all_same: bool = True,
) -> Optional[dict]:
    """Full classification pipeline for a plate string.

    Normalises the raw OCR text, validates it, identifies the format,
    and checks the dateless rule.

    Args:
        raw_text:        Unprocessed string from OCR.
        min_length:      Passed to is_plausible_uk_plate().
        max_length:      Passed to is_plausible_uk_plate().
        reject_all_same: Passed to is_plausible_uk_plate().

    Returns:
        A dict with keys {raw, normalized, format, dateless, length},
        or None if the plate is not plausible.
    """
    normalized = normalize_plate(raw_text)

    if not is_plausible_uk_plate(normalized, min_length, max_length, reject_all_same):
        return None

    # Try to identify the specific format
    fmt = matches_known_format(normalized) or "unknown"

    return {
        "raw": raw_text,           # Original OCR output, untouched
        "normalized": normalized,  # Cleaned-up version used everywhere else
        "format": fmt,             # e.g. "current", "prefix", "dateless_num_alpha"
        "dateless": is_dateless(normalized),
        "length": len(normalized),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Private helpers
# ═══════════════════════════════════════════════════════════════════════════

def _single_swap_variants(text: str) -> List[str]:
    """Generate single-character confusion variants of *text*.

    For each character that appears in CHAR_CONFUSIONS, produce a copy
    of *text* with just that one character replaced.  Always includes the
    original string.

    We deliberately limit to single swaps to avoid a combinatorial
    explosion (2^n variants) and to keep false-positive rates low.

    Example:
        "AB10" → {"AB10", "A810", "ABI0", "AB1O"}
    """
    # Using a set avoids duplicates automatically
    variants = {text}
    for i, ch in enumerate(text):
        replacement = CHAR_CONFUSIONS.get(ch)
        if replacement:
            # Build a new string with one character swapped.
            # Python strings are immutable so we slice + concatenate.
            variants.add(text[:i] + replacement + text[i + 1:])
    return list(variants)
