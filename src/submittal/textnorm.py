"""Deterministic text normalization for evidence hashing and matching.

normalize_text() does NOT alter stored raw_text — it only produces a stable
form used for hashing and later phrase matching.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

# Common Unicode dash / minus variants → ASCII hyphen-minus
_DASH_TRANS = str.maketrans(
    {
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
        "\uFE58": "-",  # small em dash
        "\uFE63": "-",  # small hyphen-minus
        "\uFF0D": "-",  # fullwidth hyphen-minus
    }
)

# Curly quotes / apostrophes → ASCII
_QUOTE_TRANS = str.maketrans(
    {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201A": "'",  # single low-9 quotation mark
        "\u201B": "'",  # single high-reversed-9 quotation mark
        "\u2032": "'",  # prime
        "\u201C": '"',  # left double quotation mark
        "\u201D": '"',  # right double quotation mark
        "\u201E": '"',  # double low-9 quotation mark
        "\u201F": '"',  # double high-reversed-9 quotation mark
        "\u2033": '"',  # double prime
        "\u00AB": '"',  # left guillemet
        "\u00BB": '"',  # right guillemet
    }
)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return a deterministic normalized form for hashing/matching.

    Preserves letters, numbers, and general punctuation/meaning.
    Does not spell-correct, does not strip punctuation, and does not
    lowercase (callers that need case-folding for match may lower separately).
    """
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.translate(_DASH_TRANS)
    value = value.translate(_QUOTE_TRANS)
    value = _WHITESPACE_RE.sub(" ", value)
    return value.strip()


def normalized_text_hash(text: str) -> str:
    """SHA-256 hex digest of normalize_text(text). Identity/change detection only."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
