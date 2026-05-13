"""
core/embedder.py
══════════════════════════════════════════════════════════════════
Latent Semantic Analysis (LSA) embedding pipeline.

Architecture decision (documented):
  We use TF-IDF → TruncatedSVD → L2-Normalizer.
  This is identical to what scikit-learn calls "LSA" and what the
  original academic RAG literature called "vector space retrieval."

  Neural embeddings (sentence-transformers) would score ~5-8%
  higher on MTEB benchmarks. For a study app:
    - Academic text is declarative and literal → smaller semantic gap
    - BM25 hybrid compensates for vocabulary mismatch
    - Zero API calls = zero latency on embedding path
    - No model download = works on air-gapped machines
    - Future upgrade: swap LSAEmbedder for JinaEmbedder — same interface

Pipeline internals:
  TfidfVectorizer:
    - max_features=15,000  (caps vocabulary, controls memory)
    - ngram_range=(1,2)    (bigrams capture "gradient descent")
    - sublinear_tf=True    (log(1+tf) prevents term dominance)

  TruncatedSVD (n_components=256):
    - Compresses sparse 15k-dim into dense 256-dim
    - Captures latent topics: "cardiac" and "heart" cluster together

  Normalizer (L2):
    - dot_product(a,b) == cosine_similarity(a,b) after normalisation

Persistence:
  Pickled pipeline, key = corpus_hash = md5(sorted(doc_hashes)).
  Same PDFs → same hash → instant cache hit.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from config.settings import (
    EMBED_DIR,
    EMBEDDING_DIM,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_MIN,
    TFIDF_NGRAM_MAX,
    MIN_DOC_FREQ,
    MAX_DOC_FREQ_RATIO,
)
from utils.logger import get_logger

log = get_logger(__name__)


class LSAEmbedder:
    """
    TF-IDF + Truncated SVD (LSA) embedding pipeline.

    Lifecycle:
      1. embedder = LSAEmbedder()
      2. embedder.fit(all_chunk_texts, corpus_hash)
      3. doc_vecs = embedder.transform(chunk_texts)   → store in VectorStore
      4. q_vec    = embedder.transform_one(query)     → search VectorStore
      5. embedder.save()  /  LSAEmbedder.load(hash)  → persistence
    """

    def __init__(self) -> None:
        self._pipeline: Optional[Pipeline] = None
        self._corpus_hash: str = ""
        self._vocab_size: int = 0
        self._n_components: int = 0
        self._fit_time_ms: int = 0
        self._n_docs: int = 0

    # ── Fit ────────────────────────────────────────────────────

    def fit(self, texts: List[str], corpus_hash: str = "") -> "LSAEmbedder":
        """
        Build vocabulary and SVD decomposition from corpus texts.

        Args:
            texts:        All chunk texts (the entire retrieval corpus).
            corpus_hash:  Stable identifier — used as disk cache key.

        Returns:
            self (fluent interface for method chaining).

        Raises:
            ValueError: corpus is empty or too small for SVD.
        """
        if not texts:
            raise ValueError("Cannot fit embedder on empty corpus.")
        n = len(texts)
        if n < 3:
            raise ValueError(
                f"Need at least 3 text chunks to build embeddings. Got {n}. "
                "Upload a larger document."
            )

        # SVD requires n_components < min(n_samples, n_features)
        n_components = min(EMBEDDING_DIM, n - 1)

        log.info("Fitting LSA: %d texts → %d dims", n, n_components)
        t0 = time.monotonic()

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=(TFIDF_NGRAM_MIN, TFIDF_NGRAM_MAX),
                min_df=MIN_DOC_FREQ,
                max_df=MAX_DOC_FREQ_RATIO,
                sublinear_tf=True,
                strip_accents="unicode",
                token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9\-\.]{1,}\b",
            )),
            ("svd", TruncatedSVD(
                n_components=n_components,
                algorithm="randomized",
                n_iter=7,
                random_state=42,
            )),
            ("norm", Normalizer(norm="l2", copy=False)),
        ])

        self._pipeline.fit(texts)
        self._corpus_hash = corpus_hash
        self._vocab_size = len(self._pipeline["tfidf"].vocabulary_)
        self._n_components = n_components
        self._fit_time_ms = int((time.monotonic() - t0) * 1000)
        self._n_docs = n

        log.info(
            "LSA fit: vocab=%d, dim=%d, n=%d, %dms",
            self._vocab_size, n_components, n, self._fit_time_ms,
        )
        return self

    # ── Transform ──────────────────────────────────────────────

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts.

        Returns:
            np.ndarray shape (len(texts), n_components), float32, L2-normalised.

        Raises:
            RuntimeError: embedder not yet fit.
        """
        self._require_fitted()
        return self._pipeline.transform(texts).astype(np.float32)

    def transform_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (1, n_components)."""
        return self.transform([text])

    # ── Persistence ────────────────────────────────────────────

    def save(self) -> Path:
        """Persist to disk. Returns path written."""
        self._require_fitted()
        path = self._cache_path(self._corpus_hash)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline":    self._pipeline,
                "corpus_hash": self._corpus_hash,
                "vocab_size":  self._vocab_size,
                "n_components": self._n_components,
                "fit_time_ms": self._fit_time_ms,
                "n_docs":      self._n_docs,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("LSAEmbedder → %s (%.1f KB)", path, path.stat().st_size / 1024)
        return path

    @classmethod
    def load(cls, corpus_hash: str) -> Optional["LSAEmbedder"]:
        """
        Load from disk cache. Returns None on miss or corrupt cache.
        """
        path = cls._cache_path(corpus_hash)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            obj = cls()
            obj._pipeline     = data["pipeline"]
            obj._corpus_hash  = data["corpus_hash"]
            obj._vocab_size   = data["vocab_size"]
            obj._n_components = data["n_components"]
            obj._fit_time_ms  = data["fit_time_ms"]
            obj._n_docs       = data["n_docs"]
            log.info("LSAEmbedder loaded: dim=%d, vocab=%d", obj._n_components, obj._vocab_size)
            return obj
        except Exception as exc:
            log.warning("Cache corrupt (%s): %s — will re-fit.", path, exc)
            path.unlink(missing_ok=True)
            return None

    # ── Helpers ────────────────────────────────────────────────

    def _require_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("LSAEmbedder not fitted. Call fit(texts) first.")

    @staticmethod
    def _cache_path(corpus_hash: str) -> Path:
        return EMBED_DIR / f"lsa_{corpus_hash or 'default'}.pkl"

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    @property
    def embedding_dim(self) -> int:
        return self._n_components

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def corpus_size(self) -> int:
        return self._n_docs

    @property
    def stats(self) -> dict:
        return {
            "fitted": self.is_fitted,
            "vocab_size": self._vocab_size,
            "embedding_dim": self._n_components,
            "corpus_docs": self._n_docs,
            "fit_time_ms": self._fit_time_ms,
        }


# ── Module-level helper ────────────────────────────────────────

def compute_corpus_hash(doc_hashes: List[str]) -> str:
    """
    Order-independent hash of a document set.
    Same PDFs (any upload order) → same hash → cache hit.
    """
    combined = "|".join(sorted(doc_hashes))
    return hashlib.md5(combined.encode()).hexdigest()[:12]
