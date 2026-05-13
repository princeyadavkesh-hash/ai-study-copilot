from dataclasses import dataclass
from typing import List
import numpy as np
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchResult:
    chunk: any
    score: float


class VectorStore:

    def __init__(self):

        self.vectors = []

        self.chunks = []

        self.document_names = set()

    def add_embeddings(
        self,
        embeddings,
        chunks
    ):

        embeddings = np.array(embeddings)

        if len(self.vectors) == 0:

            self.vectors = embeddings

        else:

            self.vectors = np.vstack([
                self.vectors,
                embeddings
            ])

        self.chunks.extend(chunks)

        for chunk in chunks:

            self.document_names.add(
                chunk.doc_name
            )

        print(
            f"Added {len(chunks)} vectors."
        )

    def search(
        self,
        query_embedding,
        top_k=4
    ):

        if len(self.chunks) == 0:

            return []

        similarities = cosine_similarity(
            [query_embedding],
            self.vectors
        )[0]

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []

        for idx in top_indices:

            results.append(
                SearchResult(
                    chunk=self.chunks[idx],
                    score=float(similarities[idx])
                )
            )

        return results

    def has_document(
        self,
        doc_name
    ):

        return doc_name in self.document_names

    def save(self):

        os.makedirs(
            "data/vector_cache",
            exist_ok=True
        )

        with open(
            "data/vector_cache/store.pkl",
            "wb"
        ) as f:

            pickle.dump(self, f)

    @staticmethod
    def load():

        path = "data/vector_cache/store.pkl"

        if not os.path.exists(path):

            return None

        with open(path, "rb") as f:

            return pickle.load(f)

    def reset(self):

        self.vectors = []

        self.chunks = []

        self.document_names = set()

        path = "data/vector_cache/store.pkl"

        if os.path.exists(path):

            os.remove(path)