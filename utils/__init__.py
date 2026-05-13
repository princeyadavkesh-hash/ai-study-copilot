from .logger import get_logger
from .text_utils import clean_pdf_text, chunk_text, normalize_query, truncate_text, excerpt
# cache_manager imported directly by consumers to avoid circular import:
#   utils.cache_manager → core.pdf_processor (TextChunk)
#   core.pdf_processor  → utils (logger, text_utils)
# Import chain is safe when cache_manager is imported directly, not via utils/__init__
