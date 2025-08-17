# src/retriever.py
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class ComplaintRetriever:
    def __init__(self, embedding_model: SentenceTransformer, index, metadata):
        self.embedding_model = embedding_model
        self.index = index
        self.metadata = metadata

    def embed_query(self, query: str) -> np.ndarray:
        return self.embedding_model.encode([query])[0]

    def retrieve_chunks(self, query: str, k: int = 5):
        query_vector = self.embed_query(query).astype("float32")
        distances, indices = self.index.search(np.array([query_vector]), k)
        return [self.metadata[idx]["text_chunk"] for idx in indices[0]]
