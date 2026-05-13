"""
utils/text_utils.py
══════════════════════════════════════════════════════════════════
Pure text-processing utilities with no external dependencies.

Why a separate module: chunker, retriever, and prompt builder all
need text normalisation and truncation. One implementation tested
once, used everywhere.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List


# ── Text Cleaning ──────────────────────────────────────────────

def clean_pdf_text(text: str) -> str:
    """
    Normalise raw PDF-extracted text for downstream processing.

    Operations (in order):
    1. Unicode normalise to NFC (handles ligatures, accented chars)
    2. Strip control characters (PDF artifacts: 0x00-0x08, 0x0e-0x1f)
    3. Collapse multiple blank lines → single blank line
    4. Collapse runs of spaces/tabs → single space
    5. Strip leading/trailing whitespace
    """
    # 1. Unicode normalise
    text = unicodedata.normalize("NFC", text)

    # 2. Remove control characters (keep \t \n \r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Collapse 3+ newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Collapse horizontal whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 5. Strip
    return text.strip()


def truncate_text(text: str, max_chars: int, suffix: str = "…") -> str:
    """
    Truncate text to max_chars, breaking at a word boundary.
    Never truncates in the middle of a word.
    """
    if len(text) <= max_chars:
        return text

    # Find last space before limit
    cutoff = text.rfind(" ", 0, max_chars - len(suffix))
    if cutoff == -1:
        cutoff = max_chars - len(suffix)

    return text[:cutoff] + suffix


def normalize_query(query: str) -> str:
    """
    Light normalisation for user queries before embedding.
    - Strip whitespace
    - Collapse internal spaces
    - Lowercase (TF-IDF handles case, but consistent input is cleaner)
    """
    return re.sub(r"\s+", " ", query).strip().lower()


# ── Chunking Helpers ────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int = 80,
) -> List[str]:
    """
    Split text into overlapping character-bounded chunks.

    Why character-based (not token-based):
      No tiktoken available without network install. Characters are
      a stable proxy: avg English token ≈ 4 chars, so chunk_size=1200
      ≈ 300 tokens, well within LLM context limits.

    Algorithm:
      Slide a window of `chunk_size` chars with `overlap` chars
      of lookback at each step. Prefer to break at paragraph or
      sentence boundaries for semantic coherence.

    Args:
        text:           Source text to chunk.
        chunk_size:     Target chars per chunk.
        overlap:        Chars of overlap between successive chunks.
        min_chunk_size: Discard chunks shorter than this.

    Returns:
        List of chunk strings.
    """
    if not text or len(text) < min_chunk_size:
        return [text] if text and len(text) >= min_chunk_size else []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Prefer to break at paragraph boundary within last 20% of window
        if end < text_len:
            search_start = max(start + int(chunk_size * 0.8), start + 1)
            para_break = text.rfind("\n\n", search_start, end)
            if para_break != -1:
                end = para_break + 2  # include the newlines

            # Fall back to sentence boundary
            elif (sent_break := _last_sentence_end(text, search_start, end)) != -1:
                end = sent_break

        chunk = text[start:end].strip()

        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        # Advance start, but keep `overlap` chars of context
        next_start = end - overlap
        if next_start <= start:
            next_start = start + max(1, chunk_size - overlap)
        start = next_start

    return chunks


def _last_sentence_end(text: str, search_start: int, search_end: int) -> int:
    """
    Find the position just after the last sentence-ending punctuation
    in text[search_start:search_end]. Returns -1 if not found.
    """
    for i in range(search_end - 1, search_start - 1, -1):
        if text[i] in ".!?" and (i + 1 >= len(text) or text[i + 1] in " \n"):
            return i + 1
    return -1


# ── Display Helpers ─────────────────────────────────────────────

def excerpt(text: str, max_chars: int = 200) -> str:
    """Return a short excerpt with ellipsis, for citation previews."""
    return truncate_text(text.replace("\n", " "), max_chars)
