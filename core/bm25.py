"""
core/bm25.py
══════════════════════════════════════════════════════════════════
Pure-Python BM25 (Best Match 25) implementation.

Why BM25 from scratch?
  rank_bm25 is not installed and we can't install it. But BM25 is
  a well-understood algorithm implementable in ~60 lines of numpy.
  This implementation matches Elasticsearch's default BM25 scorer.

BM25 formula per query term t, document d:
  score(d, t) = IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1*(1-b+b*|d|/avgdl))

  where:
    IDF(t)  = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
    tf(t,d) = raw term frequency in document d
    |d|     = document length (total terms)
    avgdl   = average document length across corpus
    k1      = 1.5  (term frequency saturation — standard value)
    b       = 0.75 (length normalisation — standard value)

Role in hybrid retrieval:
  BM25 excels at exact keyword matches: "mitochondria", "ATP synthesis",
  "p-value". LSA/TF-IDF excels at semantic similarity. Combined:
    hybrid = 0.6 * lsa_cosine + 0.4 * normalised_bm25
  This beats either method alone for academic/study text retrieval.

Persistence:
  BM25 instance is pickled alongside the LSA embedder. Same corpus
  hash → same cache → loaded together at startup.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class BM25:
    """
    Okapi BM25 scorer over a fixed corpus.

    Usage:
        bm25 = BM25()
        bm25.fit(list_of_texts)
        scores = bm25.get_scores("neural network training")  # (N,) array
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
            ngram_range=(1, 1),      # BM25 operates on unigrams
            lowercase=True,
        )
        self._idf: np.ndarray = np.array([])
        self._tf_matrix: np.ndarray = np.array([])   # (N_docs, vocab)
        self._dl: np.ndarray = np.array([])           # document lengths
        self._avgdl: float = 1.0
        self._is_fitted: bool = False

    # ── Public API ─────────────────────────────────────────────

    def fit(self, corpus: List[str]) -> "BM25":
        """
        Build BM25 index from corpus.
        Must be called before get_scores().
        """
        if not corpus:
            raise ValueError("BM25.fit() requires a non-empty corpus.")

        # Build raw term-frequency matrix (dense — OK for ≤50k docs/15k vocab)
        tf_sparse = self._vectorizer.fit_transform(corpus)
        tf_matrix = tf_sparse.toarray().astype(np.float32)  # (N, V)

        n_docs = tf_matrix.shape[0]

        # Document lengths and average length
        self._dl = tf_matrix.sum(axis=1)              # (N,)
        self._avgdl = float(self._dl.mean()) or 1.0   # guard div-by-zero

        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df = (tf_matrix > 0).sum(axis=0).astype(np.float32)  # (V,)
        self._idf = np.log(
            (n_docs - df + 0.5) / (df + 0.5) + 1.0
        )

        self._tf_matrix = tf_matrix
        self._is_fitted = True
        return self

    def get_scores(self, query: str) -> np.ndarray:
        """
        Compute BM25 relevance scores for a query against all corpus docs.

        Args:
            query: Raw query string (tokenised internally).

        Returns:
            np.ndarray of shape (N_docs,), dtype float32.
            Scores are non-negative; higher = more relevant.
            Returns zeros array if query terms are all OOV.
        """
        self._require_fitted()

        # Vectorise query — only keep in-vocabulary terms
        try:
            q_vec = self._vectorizer.transform([query]).toarray()[0]  # (V,)
        except Exception:
            return np.zeros(self._tf_matrix.shape[0], dtype=np.float32)

        query_term_indices = q_vec.nonzero()[0]
        if len(query_term_indices) == 0:
            return np.zeros(self._tf_matrix.shape[0], dtype=np.float32)

        k1, b = self.k1, self.b
        scores = np.zeros(self._tf_matrix.shape[0], dtype=np.float32)

        for idx in query_term_indices:
            tf = self._tf_matrix[:, idx]         # (N,) term freq per doc
            idf = self._idf[idx]                  # scalar
            # Length-normalised TF
            denom = tf + k1 * (1.0 - b + b * self._dl / self._avgdl)
            scores += idf * (tf * (k1 + 1.0)) / denom

        return scores

    def get_scores_normalised(self, query: str) -> np.ndarray:
        """
        BM25 scores normalised to [0, 1] for hybrid scoring.
        Returns zeros if all scores are zero.
        """
        scores = self.get_scores(query)
        max_score = scores.max()
        if max_score == 0.0:
            return scores
        return scores / max_score

    # ── State ──────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def corpus_size(self) -> int:
        return self._tf_matrix.shape[0] if self._is_fitted else 0

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("BM25 must be fit before scoring. Call fit().")
