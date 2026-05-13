"""
core/vector_store.py
══════════════════════════════════════════════════════════════════
In-memory vector store backed by numpy arrays.

Why numpy instead of FAISS?
  FAISS is not installed in this environment. numpy + sklearn's
  cosine_similarity is equally correct for our scale:
  - Up to ~50k chunks: cosine_similarity runs in <50ms (numpy BLAS)
  - Exact search (no approximation error unlike IVF-FAISS)
  - Zero additional dependencies
  - Easy to inspect, test, and debug

Architecture:
  VectorStore holds two parallel arrays:
    _vectors:   np.ndarray (N, dim) — float32 embeddings
    _chunks:    List[TextChunk]     — metadata, same row order

  Search = matrix multiply (dot product on L2-normalized vectors)
  which equals cosine similarity. sklearn handles batching.

Persistence:
  Both arrays serialised with numpy.savez_compressed (vectors)
  + pickle (chunks). Compression ratio ~3:1 for embedding matrices.

Multi-document:
  All chunks from all PDFs go into one unified index.
  Cross-document retrieval ("compare Chapter 2 from doc A with
  doc B's intro") works naturally.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import TOP_K, MIN_SIMILARITY_SCORE, VECTOR_DIR
from core.pdf_processor import TextChunk
from utils.logger import get_logger

log = get_logger(__name__)

_VECTORS_FILE = VECTOR_DIR / "vectors.npz"
_CHUNKS_FILE  = VECTOR_DIR / "chunks.pkl"


# ── Result type ────────────────────────────────────────────────

class SearchResult:
    """One retrieved chunk with its similarity score."""

    __slots__ = ("chunk", "score", "rank")

    def __init__(self, chunk: TextChunk, score: float, rank: int) -> None:
        self.chunk = chunk
        self.score = score
        self.rank = rank   # 1-indexed, 1 = most relevant

    def __repr__(self) -> str:
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.3f}, "
            f"doc={self.chunk.doc_name!r}, page={self.chunk.page_num})"
        )


# ── VectorStore ────────────────────────────────────────────────

class VectorStore:
    """
    Unified vector index for all uploaded documents.

    Designed to live in Flask's app-level state (one per server
    process). Thread safety note: add_embeddings and search are
    read-mostly; use a lock for concurrent writes if scaling to
    multi-threaded prod (gunicorn workers each get their own copy).
    """

    def __init__(self) -> None:
        self._vectors: Optional[np.ndarray] = None  # (N, dim) float32
        self._chunks: List[TextChunk] = []
        self._doc_names: set[str] = set()

    # ── Ingestion ──────────────────────────────────────────────

    def add_embeddings(
        self,
        vectors: np.ndarray,
        chunks: List[TextChunk],
    ) -> None:
        """
        Add a batch of (vector, chunk) pairs to the store.

        Args:
            vectors: Shape (N, dim), float32, already L2-normalised.
            chunks:  Parallel list of TextChunk metadata.

        The caller is responsible for ensuring doc deduplication
        before calling this. Check `has_document(doc_name)` first.
        """
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) != chunk count ({len(chunks)})"
            )

        vectors = vectors.astype(np.float32)

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

        self._chunks.extend(chunks)

        for chunk in chunks:
            self._doc_names.add(chunk.doc_name)

        log.info(
            "Added %d vectors. Store total: %d (docs: %s)",
            len(chunks),
            self.total_chunks,
            ", ".join(sorted(self._doc_names)),
        )

    # ── Search ─────────────────────────────────────────────────

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = TOP_K,
        min_score: float = MIN_SIMILARITY_SCORE,
    ) -> List[SearchResult]:
        """
        Find the top-k most semantically similar chunks.

        Args:
            query_vector: Shape (1, dim) or (dim,), float32.
            top_k:        Maximum results to return.
            min_score:    Minimum cosine similarity threshold.

        Returns:
            List of SearchResult, sorted by score descending.
            May be shorter than top_k if few chunks score above min_score.
        """
        if self.is_empty:
            log.warning("Search called on empty vector store.")
            return []

        query_vector = query_vector.astype(np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        t0 = time.monotonic()

        # cosine_similarity returns (1, N) array
        scores: np.ndarray = cosine_similarity(query_vector, self._vectors)[0]

        # Get top-k indices (unsorted), then sort
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]          # O(N)
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]  # sort

        results: List[SearchResult] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score < min_score:
                break   # sorted descending — no more above threshold
            results.append(SearchResult(
                chunk=self._chunks[idx],
                score=score,
                rank=rank,
            ))

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        log.debug(
            "Search: %d results (threshold=%.2f) in %dms",
            len(results), min_score, elapsed_ms,
        )
        return results

    # ── Deduplication ──────────────────────────────────────────

    def has_document(self, doc_name: str) -> bool:
        """True if this document's chunks are already indexed."""
        return doc_name in self._doc_names

    # ── Persistence ────────────────────────────────────────────

    def save(self) -> None:
        """
        Persist vectors + chunk metadata to disk.

        Uses:
          numpy.savez_compressed for vectors (3:1 compression)
          pickle for chunks (arbitrary Python objects)
        """
        if self.is_empty:
            log.warning("Save called on empty store — nothing written.")
            return

        np.savez_compressed(str(_VECTORS_FILE), vectors=self._vectors)

        with open(_CHUNKS_FILE, "wb") as f:
            pickle.dump(
                {
                    "chunks": self._chunks,
                    "doc_names": list(self._doc_names),
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        log.info(
            "VectorStore saved: %d vectors → %s",
            self.total_chunks, _VECTORS_FILE,
        )

    @classmethod
    def load(cls) -> Optional["VectorStore"]:
        """
        Load a persisted vector store from disk.
        Returns None if no saved state exists.
        """
        if not _VECTORS_FILE.exists() or not _CHUNKS_FILE.exists():
            return None
        try:
            store = cls()
            data = np.load(str(_VECTORS_FILE))
            store._vectors = data["vectors"].astype(np.float32)

            with open(_CHUNKS_FILE, "rb") as f:
                meta = pickle.load(f)
            store._chunks = meta["chunks"]
            store._doc_names = set(meta["doc_names"])

            log.info(
                "VectorStore loaded: %d vectors, docs: %s",
                store.total_chunks,
                ", ".join(sorted(store._doc_names)),
            )
            return store
        except Exception as exc:
            log.error("VectorStore load failed: %s", exc)
            _VECTORS_FILE.unlink(missing_ok=True)
            _CHUNKS_FILE.unlink(missing_ok=True)
            return None

    def reset(self) -> None:
        """Wipe all data and remove persisted files."""
        self._vectors = None
        self._chunks = []
        self._doc_names = set()
        _VECTORS_FILE.unlink(missing_ok=True)
        _CHUNKS_FILE.unlink(missing_ok=True)
        log.info("VectorStore reset.")

    # ── Properties ─────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return self._vectors is None or len(self._chunks) == 0

    @property
    def document_names(self) -> List[str]:
        return sorted(self._doc_names)

    def __repr__(self) -> str:
        return (
            f"VectorStore(chunks={self.total_chunks}, "
            f"docs={list(self._doc_names)})"
        )
