import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LSAEmbedder:

    def __init__(self, n_components=100):

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )

        self.is_fitted = False

    def fit(self, texts):

        if len(texts) == 0:
            return

        tfidf_matrix = self.vectorizer.fit_transform(
            texts
        )

        max_components = min(
            tfidf_matrix.shape[0] - 1,
            tfidf_matrix.shape[1] - 1,
            100
        )

        if max_components < 2:
            max_components = 2

        self.svd = TruncatedSVD(
            n_components=max_components,
            random_state=42
        )

        self.svd.fit(tfidf_matrix)

        self.is_fitted = True

    def transform(self, texts):

        tfidf_matrix = self.vectorizer.transform(
            texts
        )

        return self.svd.transform(
            tfidf_matrix
        )

    def transform_one(self, text):

        return self.transform([text])[0]