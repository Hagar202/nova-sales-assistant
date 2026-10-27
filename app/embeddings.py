"""Embedding models: the real sentence-transformers model, and a light offline stand-in."""

import hashlib
import re

import numpy as np


class HashEmbeddings:
    """
    Offline stand-in for tests and mock mode: a hashed bag of words.

    It needs no download and no GPU, and unlike random fake embeddings, texts that
    share words get similar vectors, so retrieval in tests still behaves sensibly.
    """

    def __init__(self, size=384):
        self.size = size

    def _embed(self, text):
        vector = np.zeros(self.size, dtype="float32")
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "little") % self.size] += 1.0
        return vector.tolist()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


class SentenceTransformerEmbeddings:
    """all-MiniLM-L6-v2 (or any sentence-transformers model). Small enough to run on a CPU."""

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(list(texts), normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()


def load_embeddings(model_name, mock=False):
    if mock:
        return HashEmbeddings()
    return SentenceTransformerEmbeddings(model_name)
