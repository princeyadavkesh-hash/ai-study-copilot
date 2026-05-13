from .pdf_processor import process_pdf, ParsedDocument, TextChunk
from .embedder import LSAEmbedder, compute_corpus_hash
from .bm25 import BM25
# VectorStore and retriever imported directly by consumers (avoid circular deps)
