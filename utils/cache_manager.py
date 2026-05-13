"""
utils/cache_manager.py
══════════════════════════════════════════════════════════════════
Atomic persistence layer for the complete retrieval index.

Problem this solves:
  The retrieval index has four interdependent components that must
  always be in sync with each other:

    LSAEmbedder  — TF-IDF vocab + SVD weights
    BM25         — term frequency matrix + IDF weights
    vectors      — np.ndarray (N, dim) of chunk embeddings
    chunks       — List[TextChunk] parallel metadata

  If any one of these is stale or missing, retrieval breaks. A naive
  approach (saving each independently) creates a TOCTOU window where
  the app can load a new embedder but old vectors, causing silent
  dimension mismatches and garbage results.

  CacheManager treats all four as one atomic unit:
    - Save: write all four; only mark complete with a manifest file
    - Load: verify manifest before loading anything
    - Validation: check corpus_hash, vector shape, chunk count parity

Corpus hash as cache key:
  corpus_hash = md5(sorted(doc_file_hashes))
  Same set of PDFs → same hash → cache hit → instant startup.
  Adding one new PDF → new hash → full rebuild (correct behaviour,
  because the embedder must be re-fit on the full corpus).

Concurrency:
  Flask dev server is single-threaded; no locking needed here.
  For gunicorn multi-worker: use --preload so workers share one
  process that calls load_or_build(), then fork.

Cache directory layout:
  data/vector_cache/
    {corpus_hash}_vectors.npz     compressed float32 matrix
    {corpus_hash}_chunks.pkl      List[TextChunk]
    {corpus_hash}_bm25.pkl        BM25 instance
    {corpus_hash}_manifest.json   metadata + integrity check

  data/embeddings_cache/
    lsa_{corpus_hash}.pkl         LSAEmbedder pipeline

  Stale entries (different hash) are pruned on each save to keep
  the cache directory from growing unboundedly.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config.settings import VECTOR_DIR, EMBED_DIR
from core.bm25 import BM25
from core.embedder import LSAEmbedder
from core.pdf_processor import TextChunk
from utils.logger import get_logger

log = get_logger(__name__)


# ── Index dataclass ────────────────────────────────────────────

@dataclass
class RetrievalIndex:
    """
    The complete, self-consistent retrieval index.

    Holds every component needed for hybrid search. Created once
    after fitting and passed around by the Flask application.

    Attributes:
        embedder:     Fitted LSAEmbedder — transforms text to vectors.
        bm25:         Fitted BM25 — scores keyword relevance.
        vectors:      np.ndarray (N, dim) — all chunk embeddings.
        chunks:       List[TextChunk] — parallel metadata (same order).
        corpus_hash:  Identifies which document set produced this index.
        doc_names:    Sorted list of indexed document names.
        built_at:     Unix timestamp of when the index was built.
        build_ms:     Total time to build (0 if loaded from cache).
    """
    embedder:    LSAEmbedder
    bm25:        BM25
    vectors:     np.ndarray      # float32, L2-normalised, (N, dim)
    chunks:      List[TextChunk]
    corpus_hash: str
    doc_names:   List[str]
    built_at:    float
    build_ms:    int

    # ── Convenience properties ─────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def embedding_dim(self) -> int:
        return self.vectors.shape[1] if self.vectors.ndim == 2 else 0

    @property
    def is_consistent(self) -> bool:
        """Sanity check: all components agree on corpus size."""
        return (
            len(self.chunks) == self.vectors.shape[0]
            and self.bm25.corpus_size == len(self.chunks)
            and self.embedder.is_fitted
        )

    def stats(self) -> dict:
        return {
            "total_chunks":  self.total_chunks,
            "embedding_dim": self.embedding_dim,
            "vocab_size":    self.embedder.vocab_size,
            "corpus_hash":   self.corpus_hash,
            "doc_names":     self.doc_names,
            "built_at":      self.built_at,
            "build_ms":      self.build_ms,
            "consistent":    self.is_consistent,
        }


# ── CacheManager ───────────────────────────────────────────────

class CacheManager:
    """
    Atomic save/load for the full RetrievalIndex.

    All public methods are classmethods — no instance needed.
    The app calls:
        index = CacheManager.load(corpus_hash)   # try cache
        if index is None:
            index = CacheManager.build(chunks, corpus_hash)
            CacheManager.save(index)
    """

    # ── Save ───────────────────────────────────────────────────

    @classmethod
    def save(cls, index: RetrievalIndex) -> bool:
        """
        Atomically persist all four index components to disk.

        Write order:
          1. vectors.npz   (largest, write first to fail fast)
          2. chunks.pkl
          3. bm25.pkl
          4. lsa_*.pkl     (via embedder.save())
          5. manifest.json (written LAST — marks cache as complete)

        If any step fails, the manifest is not written, so a
        subsequent load() will see an incomplete cache and return None.

        Args:
            index: Fully-built RetrievalIndex to persist.

        Returns:
            True on success, False on any error.
        """
        h = index.corpus_hash
        log.info("Saving index (hash=%s, chunks=%d)...", h, index.total_chunks)
        t0 = time.monotonic()

        try:
            # 1. Vectors
            vec_path = cls._vec_path(h)
            np.savez_compressed(str(vec_path), vectors=index.vectors)
            log.debug("Vectors saved: %s (%.1f KB)", vec_path, vec_path.stat().st_size / 1024)

            # 2. Chunks
            chunk_path = cls._chunk_path(h)
            with open(chunk_path, "wb") as f:
                pickle.dump(index.chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.debug("Chunks saved: %s", chunk_path)

            # 3. BM25
            bm25_path = cls._bm25_path(h)
            with open(bm25_path, "wb") as f:
                pickle.dump(index.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.debug("BM25 saved: %s", bm25_path)

            # 4. LSAEmbedder (writes to EMBED_DIR with its own naming)
            index.embedder.save()

            # 5. Manifest — written LAST to signal completeness
            manifest = {
                "corpus_hash":  h,
                "total_chunks": index.total_chunks,
                "embedding_dim": index.embedding_dim,
                "doc_names":    index.doc_names,
                "built_at":     index.built_at,
                "build_ms":     index.build_ms,
                "saved_at":     time.time(),
                "version":      "1",
            }
            cls._write_manifest(h, manifest)

            elapsed = int((time.monotonic() - t0) * 1000)
            log.info("Index saved in %dms.", elapsed)

            # Prune stale caches (different hashes)
            cls._prune_stale(current_hash=h)
            return True

        except Exception as exc:
            log.error("Failed to save index: %s", exc, exc_info=True)
            # Remove partial files to avoid corrupt state
            cls._delete_artifacts(h)
            return False

    # ── Load ───────────────────────────────────────────────────

    @classmethod
    def load(cls, corpus_hash: str) -> Optional[RetrievalIndex]:
        """
        Load a cached index if one exists and is valid.

        Validation steps:
          1. Manifest exists and has matching hash
          2. All artifact files exist
          3. Vector shape matches manifest chunk count
          4. Chunk list length matches manifest

        Returns:
            RetrievalIndex on success, None on any failure.
            None is safe — the caller will rebuild from scratch.
        """
        log.info("Checking cache for hash=%s...", corpus_hash)

        # 1. Manifest check
        manifest = cls._read_manifest(corpus_hash)
        if manifest is None:
            log.info("Cache miss: no manifest for %s", corpus_hash)
            return None
        if manifest.get("corpus_hash") != corpus_hash:
            log.warning("Manifest hash mismatch. Stale cache.")
            return None

        # 2. All files must exist
        required = [
            cls._vec_path(corpus_hash),
            cls._chunk_path(corpus_hash),
            cls._bm25_path(corpus_hash),
        ]
        for p in required:
            if not p.exists():
                log.warning("Cache incomplete: missing %s", p)
                return None

        t0 = time.monotonic()
        try:
            # 3. Load vectors
            data = np.load(str(cls._vec_path(corpus_hash)))
            vectors: np.ndarray = data["vectors"].astype(np.float32)

            # 4. Load chunks
            with open(cls._chunk_path(corpus_hash), "rb") as f:
                chunks: List[TextChunk] = pickle.load(f)

            # 5. Shape validation
            expected_n = manifest["total_chunks"]
            if vectors.shape[0] != expected_n or len(chunks) != expected_n:
                log.error(
                    "Shape mismatch: manifest=%d, vectors=%d, chunks=%d",
                    expected_n, vectors.shape[0], len(chunks),
                )
                return None

            # 6. Load BM25
            with open(cls._bm25_path(corpus_hash), "rb") as f:
                bm25: BM25 = pickle.load(f)

            # 7. Load LSAEmbedder
            embedder = LSAEmbedder.load(corpus_hash)
            if embedder is None:
                log.warning("Embedder cache missing for %s", corpus_hash)
                return None

            elapsed = int((time.monotonic() - t0) * 1000)
            log.info(
                "Cache HIT: %d chunks, dim=%d, loaded in %dms",
                len(chunks), vectors.shape[1], elapsed,
            )

            return RetrievalIndex(
                embedder=embedder,
                bm25=bm25,
                vectors=vectors,
                chunks=chunks,
                corpus_hash=corpus_hash,
                doc_names=manifest.get("doc_names", []),
                built_at=manifest.get("built_at", 0.0),
                build_ms=0,   # loaded, not rebuilt
            )

        except Exception as exc:
            log.error("Cache load failed: %s", exc, exc_info=True)
            cls._delete_artifacts(corpus_hash)
            return None

    # ── Build ───────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        chunks: List[TextChunk],
        corpus_hash: str,
        doc_names: Optional[List[str]] = None,
    ) -> RetrievalIndex:
        """
        Build a complete RetrievalIndex from scratch.

        Fitting order matters:
          1. Collect all chunk texts
          2. Fit LSAEmbedder on all texts (builds vocab + SVD)
          3. Transform all texts → dense vectors
          4. Fit BM25 on all texts (builds TF matrix)
          Both models must see the same corpus in the same order.

        Args:
            chunks:      All TextChunk objects from all loaded PDFs.
            corpus_hash: Stable identifier for this document set.
            doc_names:   Display names of source documents.

        Returns:
            Fully-populated RetrievalIndex.

        Raises:
            ValueError: chunks list is empty or too small.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")
        if len(chunks) < 3:
            raise ValueError(
                f"Need at least 3 chunks to build a retrieval index. "
                f"Got {len(chunks)}. Upload a larger document."
            )

        log.info("Building index from scratch: %d chunks...", len(chunks))
        t0 = time.monotonic()

        texts = [c.text for c in chunks]

        # ── Fit LSA embedder ──────────────────────────────────
        embedder = LSAEmbedder()
        embedder.fit(texts, corpus_hash=corpus_hash)

        # ── Produce dense vectors ─────────────────────────────
        vectors = embedder.transform(texts)   # (N, dim), float32, L2-norm

        # ── Fit BM25 ──────────────────────────────────────────
        bm25 = BM25()
        bm25.fit(texts)

        # ── Resolve doc names ─────────────────────────────────
        if doc_names is None:
            doc_names = sorted({c.doc_name for c in chunks})

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        log.info(
            "Index built: chunks=%d, dim=%d, vocab=%d, bm25_terms=%d, %dms",
            len(chunks), embedder.embedding_dim,
            embedder.vocab_size, bm25.corpus_size, elapsed_ms,
        )

        return RetrievalIndex(
            embedder=embedder,
            bm25=bm25,
            vectors=vectors,
            chunks=chunks,
            corpus_hash=corpus_hash,
            doc_names=doc_names,
            built_at=time.time(),
            build_ms=elapsed_ms,
        )

    # ── Existence check ────────────────────────────────────────

    @classmethod
    def exists(cls, corpus_hash: str) -> bool:
        """Return True if a complete cache exists for this hash."""
        manifest = cls._read_manifest(corpus_hash)
        if manifest is None:
            return False
        required = [
            cls._vec_path(corpus_hash),
            cls._chunk_path(corpus_hash),
            cls._bm25_path(corpus_hash),
        ]
        return all(p.exists() for p in required)

    # ── Wipe ──────────────────────────────────────────────────

    @classmethod
    def clear_all(cls) -> None:
        """
        Delete all cached indexes (both VECTOR_DIR and EMBED_DIR).
        Called when the user clicks Reset in the UI.
        """
        count = 0
        for pattern in ["*.npz", "*.pkl", "*.json"]:
            for f in VECTOR_DIR.glob(pattern):
                f.unlink(missing_ok=True)
                count += 1
        for f in EMBED_DIR.glob("lsa_*.pkl"):
            f.unlink(missing_ok=True)
            count += 1
        log.info("Cache cleared: %d files removed.", count)

    # ── Internal helpers ───────────────────────────────────────

    @staticmethod
    def _vec_path(h: str) -> Path:
        return VECTOR_DIR / f"{h}_vectors.npz"

    @staticmethod
    def _chunk_path(h: str) -> Path:
        return VECTOR_DIR / f"{h}_chunks.pkl"

    @staticmethod
    def _bm25_path(h: str) -> Path:
        return VECTOR_DIR / f"{h}_bm25.pkl"

    @staticmethod
    def _manifest_path(h: str) -> Path:
        return VECTOR_DIR / f"{h}_manifest.json"

    @classmethod
    def _write_manifest(cls, h: str, data: dict) -> None:
        with open(cls._manifest_path(h), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def _read_manifest(cls, h: str) -> Optional[dict]:
        path = cls._manifest_path(h)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            log.warning("Corrupt manifest at %s: %s", path, exc)
            path.unlink(missing_ok=True)
            return None

    @classmethod
    def _delete_artifacts(cls, h: str) -> None:
        """Remove all artifacts for a given hash (cleanup on failure)."""
        for p in [
            cls._vec_path(h),
            cls._chunk_path(h),
            cls._bm25_path(h),
            cls._manifest_path(h),
            EMBED_DIR / f"lsa_{h}.pkl",
        ]:
            p.unlink(missing_ok=True)

    @classmethod
    def _prune_stale(cls, current_hash: str) -> None:
        """
        Remove cached indexes for document sets other than current.
        Prevents unbounded growth when users repeatedly upload new PDFs.
        Keeps only the current hash's artifacts.
        """
        removed = 0
        for f in VECTOR_DIR.iterdir():
            if f.is_file() and not f.name.startswith(current_hash):
                f.unlink(missing_ok=True)
                removed += 1
        for f in EMBED_DIR.glob("lsa_*.pkl"):
            if not f.name.startswith(f"lsa_{current_hash}"):
                f.unlink(missing_ok=True)
                removed += 1
        if removed:
            log.info("Pruned %d stale cache file(s).", removed)
