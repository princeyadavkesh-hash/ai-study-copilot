"""
core/pdf_processor.py
══════════════════════════════════════════════════════════════════
PDF parsing and semantic chunking pipeline.

Architecture:
  ParsedDocument ← process_pdf(file_path)
        │
        ├── pages: List[PageData]   (raw per-page text + metadata)
        └── chunks: List[TextChunk] (overlapping semantic units)

Parser backends (tried in order):
  1. pdfplumber  — best layout handling; tables, columns, borders
  2. pypdf       — fast, lightweight fallback for simple PDFs

Chunking strategy:
  Character-based sliding window with semantic boundary detection.
  Prefer paragraph breaks > sentence breaks > hard cuts.
  Each chunk carries its source document + page number so the UI
  can render exact citation cards ("Biology 101.pdf · Page 7").

Design rationale for chunk size (1200 chars ≈ 300 tokens):
  - Small enough for precise retrieval (avoids diluting relevance)
  - Large enough for coherent context (avoids half-answers)
  - Overlap (200 chars) prevents concepts from being split in half
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config.settings import (
    CHUNK_SIZE_CHARS,
    CHUNK_OVERLAP_CHARS,
    CHUNK_MIN_CHARS,
)
from utils.logger import get_logger
from utils.text_utils import clean_pdf_text, chunk_text, excerpt

log = get_logger(__name__)


# ── Data Models ────────────────────────────────────────────────

@dataclass
class PageData:
    """Raw text extracted from one PDF page."""
    page_num: int        # 1-indexed (human-readable)
    text: str
    char_count: int = 0

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class TextChunk:
    """
    One semantic unit of text with full provenance.

    Fields used downstream:
      text          → embedded and sent to LLM as context
      doc_name      → display in citation cards
      page_num      → display in citation cards
      chunk_index   → used as FAISS/numpy row index
      doc_hash      → cache key for embedding persistence
    """
    text: str
    doc_name: str
    page_num: int
    chunk_index: int         # global position across all loaded docs
    doc_hash: str            # SHA256[:16] of source file
    char_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    @property
    def citation_label(self) -> str:
        """Human-readable citation string for UI display."""
        return f"{self.doc_name} · Page {self.page_num}"

    @property
    def preview(self) -> str:
        """Short excerpt for citation card body."""
        return excerpt(self.text, max_chars=180)


@dataclass
class ParsedDocument:
    """Complete result of processing one PDF file."""
    doc_name: str
    file_path: str
    file_hash: str           # used as cache key
    pages: List[PageData]
    chunks: List[TextChunk]
    total_pages: int
    total_chunks: int
    parse_time_ms: int
    parser_used: str         # "pdfplumber" or "pypdf"

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    def summary_line(self) -> str:
        return (
            f"{self.doc_name}: {self.total_pages} pages, "
            f"{self.total_chunks} chunks, "
            f"{self.total_chars:,} chars "
            f"({self.parser_used}, {self.parse_time_ms}ms)"
        )


# ── Public API ─────────────────────────────────────────────────

def process_pdf(
    file_path: str | Path,
    doc_name: Optional[str] = None,
    chunk_index_offset: int = 0,
) -> ParsedDocument:
    """
    Parse a PDF and return a fully chunked ParsedDocument.

    Args:
        file_path:           Path to the PDF file.
        doc_name:            Display name (defaults to filename stem).
        chunk_index_offset:  Start chunk_index from this value.
                             Use when loading multiple PDFs so chunk
                             indices remain globally unique.

    Returns:
        ParsedDocument with pages and chunks populated.

    Raises:
        ValueError: If the file does not exist or is not a PDF.
        RuntimeError: If both parsers fail to extract any text.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

    doc_name = doc_name or file_path.stem
    file_hash = _hash_file(file_path)

    log.info("Parsing PDF: %s (hash=%s)", doc_name, file_hash)
    t0 = time.monotonic()

    # Try parsers in order of preference
    pages, parser_used = _extract_pages(file_path, doc_name)

    if not pages:
        raise RuntimeError(
            f"Both PDF parsers failed to extract text from '{doc_name}'. "
            "The file may be scanned/image-only. Try OCR pre-processing."
        )

    # Chunk all pages
    chunks = _build_chunks(pages, doc_name, file_hash, chunk_index_offset)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    doc = ParsedDocument(
        doc_name=doc_name,
        file_path=str(file_path),
        file_hash=file_hash,
        pages=pages,
        chunks=chunks,
        total_pages=len(pages),
        total_chunks=len(chunks),
        parse_time_ms=elapsed_ms,
        parser_used=parser_used,
    )

    log.info(doc.summary_line())
    return doc


# ── Parsers ────────────────────────────────────────────────────

def _extract_pages(
    file_path: Path,
    doc_name: str,
) -> tuple[List[PageData], str]:
    """
    Try pdfplumber first; fall back to pypdf.
    Returns (pages, parser_name).
    """
    # ── pdfplumber (primary) ──
    try:
        import pdfplumber
        pages = _parse_pdfplumber(file_path)
        if pages:
            log.debug("pdfplumber: %d pages extracted", len(pages))
            return pages, "pdfplumber"
        log.warning("pdfplumber returned 0 text pages for %s", doc_name)
    except Exception as exc:
        log.warning("pdfplumber failed for %s: %s", doc_name, exc)

    # ── pypdf (fallback) ──
    try:
        import pypdf
        pages = _parse_pypdf(file_path)
        if pages:
            log.debug("pypdf: %d pages extracted", len(pages))
            return pages, "pypdf"
        log.warning("pypdf returned 0 text pages for %s", doc_name)
    except Exception as exc:
        log.warning("pypdf failed for %s: %s", doc_name, exc)

    return [], "none"


def _parse_pdfplumber(file_path: Path) -> List[PageData]:
    """Extract text per page using pdfplumber."""
    import pdfplumber

    pages: List[PageData] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            text = clean_pdf_text(raw)
            if text:
                pages.append(PageData(page_num=i, text=text))
    return pages


def _parse_pypdf(file_path: Path) -> List[PageData]:
    """Extract text per page using pypdf."""
    import pypdf

    pages: List[PageData] = []
    reader = pypdf.PdfReader(str(file_path))
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_pdf_text(raw)
        if text:
            pages.append(PageData(page_num=i, text=text))
    return pages


# ── Chunking ───────────────────────────────────────────────────

def _build_chunks(
    pages: List[PageData],
    doc_name: str,
    file_hash: str,
    index_offset: int,
) -> List[TextChunk]:
    """
    Split every page's text into overlapping chunks.
    Returns globally-indexed TextChunk list.
    """
    chunks: List[TextChunk] = []
    global_idx = index_offset

    for page in pages:
        page_chunks = chunk_text(
            page.text,
            chunk_size=CHUNK_SIZE_CHARS,
            overlap=CHUNK_OVERLAP_CHARS,
            min_chunk_size=CHUNK_MIN_CHARS,
        )

        for chunk_text_str in page_chunks:
            chunks.append(TextChunk(
                text=chunk_text_str,
                doc_name=doc_name,
                page_num=page.page_num,
                chunk_index=global_idx,
                doc_hash=file_hash,
            ))
            global_idx += 1

    log.debug(
        "Chunked %s: %d pages → %d chunks",
        doc_name, len(pages), len(chunks),
    )
    return chunks


# ── Utilities ──────────────────────────────────────────────────

def _hash_file(file_path: Path) -> str:
    """
    First 16 hex chars of SHA256. Used as cache key.
    Read in 64KB blocks to handle large PDFs without RAM spikes.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]
