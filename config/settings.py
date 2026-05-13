"""
config/settings.py
══════════════════════════════════════════════════════════════════
Central configuration hub. Every tunable constant lives here.

Architecture rationale:
  Single source of truth means a team member changes CHUNK_SIZE
  once and it propagates to chunker, retriever, and tests.
  Config-driven architecture is the first mark of production code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Directory Layout ───────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data"
UPLOAD_DIR    = DATA_DIR / "uploads"
VECTOR_DIR    = DATA_DIR / "vector_cache"
EMBED_DIR     = DATA_DIR / "embeddings_cache"
LOG_DIR       = BASE_DIR / "logs"

# Create all data dirs on import (safe, idempotent)
for _d in [UPLOAD_DIR, VECTOR_DIR, EMBED_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── LLM ────────────────────────────────────────────────────────
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY    = os.getenv("OPENROUTER_API_KEY", "")

GROQ_BASE_URL         = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_BASE_URL   = "https://openrouter.ai/api/v1/chat/completions"

# Free-tier models in preference order
GROQ_MODEL_PRIMARY    = "llama-3.3-70b-versatile"
GROQ_MODEL_FALLBACK   = "llama3-8b-8192"
OPENROUTER_MODEL      = "mistralai/mistral-7b-instruct:free"

LLM_MAX_TOKENS        = 1536
LLM_TEMPERATURE       = 0.3        # factual > creative for study use
LLM_TIMEOUT_SEC       = 45         # generous timeout for free tiers
LLM_RETRY_ATTEMPTS    = 3
LLM_RETRY_BACKOFF     = 2.0        # seconds, doubles each retry

# ── Embeddings ─────────────────────────────────────────────────
# TF-IDF + Truncated SVD (LSA) — fully local, zero cost, no GPU.
# Produces dense semantic vectors that outperform raw TF-IDF for
# synonymy and latent topic matching (LSA was the original RAG).
EMBEDDING_DIM         = 256        # SVD components; sweet spot quality/speed
TFIDF_MAX_FEATURES    = 15_000     # vocabulary cap
TFIDF_NGRAM_MIN       = 1
TFIDF_NGRAM_MAX       = 2          # unigrams + bigrams = richer semantics
MIN_DOC_FREQ          = 1          # keep rare technical terms
MAX_DOC_FREQ_RATIO    = 0.95       # drop near-universal stop words

# ── Chunking ───────────────────────────────────────────────────
CHUNK_SIZE_CHARS      = 1200       # characters per chunk (~300 tokens)
CHUNK_OVERLAP_CHARS   = 200        # overlap preserves cross-boundary context
CHUNK_MIN_CHARS       = 80         # discard micro-chunks (headers, footers)

# ── Retrieval ──────────────────────────────────────────────────
TOP_K                 = 6          # retrieve 6, display top sources
MIN_SIMILARITY_SCORE  = 0.10       # cosine threshold (LSA scores are lower than dense)
MAX_CONTEXT_CHARS     = 6000       # hard cap on context sent to LLM

# ── Memory ─────────────────────────────────────────────────────
MEMORY_MAX_TURNS      = 8          # keep last 8 user+assistant pairs
MEMORY_MAX_CHARS      = 4000       # hard cap on serialized history

# ── Flask App ──────────────────────────────────────────────────
APP_TITLE             = os.getenv("APP_TITLE", "AI Study Copilot")
APP_VERSION           = "1.0.0"
SECRET_KEY            = os.getenv("SECRET_KEY", os.urandom(32).hex())
DEBUG                 = os.getenv("DEBUG", "false").lower() == "true"
PORT                  = int(os.getenv("PORT", "5000"))
MAX_UPLOAD_MB         = 50
ALLOWED_EXTENSIONS    = {".pdf"}

# ── Study Modes ─────────────────────────────────────────────────
STUDY_MODES = {
    "qa":        {"label": "Q&A",       "icon": "💬"},
    "summarize": {"label": "Summarize", "icon": "📝"},
    "explain":   {"label": "Explain",   "icon": "🔍"},
    "quiz":      {"label": "Quiz Me",   "icon": "🧠"},
}
