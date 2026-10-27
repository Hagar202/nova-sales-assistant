"""
A small FAISS index over LangChain Documents.

We use FAISS directly (cosine similarity on normalized vectors) instead of the
langchain-community wrapper, which is being sunset. Search returns the score too,
so the assistant can see how relevant each chunk is.
"""

import faiss
import numpy as np


class VectorIndex:
    def __init__(self, documents, embeddings):
        self.documents = list(documents)
        self.embeddings = embeddings
        vectors = self._normalize(embeddings.embed_documents([d.page_content for d in self.documents]))
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product of unit vectors = cosine similarity
        self.index.add(vectors)

    @staticmethod
    def _normalize(vectors):
        array = np.asarray(vectors, dtype="float32")
        if array.ndim == 1:
            array = array[None, :]
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms

    def search(self, query, k=4):
        """Return [(document, cosine_score)] for the k most similar documents."""
        if not self.documents:
            return []
        query_vector = self._normalize(self.embeddings.embed_query(query))
        scores, ids = self.index.search(query_vector, min(k, len(self.documents)))
        return [(self.documents[i], float(s)) for s, i in zip(scores[0], ids[0], strict=False) if i >= 0]

    def __len__(self):
        return len(self.documents)
