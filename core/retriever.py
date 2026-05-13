from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.vector_store import VectorStore


@dataclass
class RetrievalResult:
    context: str
    citations: List[Dict[str, Any]]
    found: bool


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def query(self, query_vector: np.ndarray, top_k: int = 3) -> RetrievalResult:
        if len(self.vector_store.documents) == 0:
            return RetrievalResult(
                context="",
                citations=[],
                found=False
            )

        similarities = cosine_similarity(
            [query_vector],
            self.vector_store.vectors
        )[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        contexts = []
        citations = []

        for idx in top_indices:
            doc = self.vector_store.documents[idx]

            contexts.append(doc["text"])

            citations.append({
                "source": doc.get("source", "Unknown"),
                "score": float(similarities[idx])
            })

        context = "\n\n".join(contexts)

        return RetrievalResult(
            context=context,
            citations=citations,
            found=True
        )