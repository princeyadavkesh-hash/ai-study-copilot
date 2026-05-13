from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class LightweightEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=3000
        )
        self.matrix = None
        self.texts = []

    def fit(self, texts):
        self.texts = texts

        if not texts:
            self.matrix = None
            return

        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=5):
        if self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.matrix
        )[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []

        for idx in top_indices:
            results.append({
                "index": int(idx),
                "score": float(similarities[idx])
            })

        return results