from dataclasses import dataclass
from typing import List, Dict, Any

from core.vector_store import VectorStore
from core.embedder import LightweightEmbedder


@dataclass
class RetrievalResult:
    context: str
    citations: List[Dict[str, Any]]
    found: bool


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: LightweightEmbedder
    ):
        self.vector_store = vector_store
        self.embedder = embedder

    def query(self, query: str, top_k: int = 3) -> RetrievalResult:

        if len(self.vector_store.documents) == 0:
            return RetrievalResult(
                context="",
                citations=[],
                found=False
            )

        results = self.embedder.search(query, top_k=top_k)

        if not results:
            return RetrievalResult(
                context="",
                citations=[],
                found=False
            )

        contexts = []
        citations = []

        for result in results:
            idx = result["index"]

            if idx >= len(self.vector_store.documents):
                continue

            doc = self.vector_store.documents[idx]

            contexts.append(doc["text"])

            citations.append({
                "source": doc.get("source", "Unknown"),
                "score": round(result["score"], 4)
            })

        context = "\n\n".join(contexts)

        return RetrievalResult(
            context=context,
            citations=citations,
            found=True
        )