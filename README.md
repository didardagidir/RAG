"""Dense semantic index using sentence-transformers embeddings + Chroma."""
from __future__ import annotations

from src.config import CONFIG, ENV


class VectorIndex:
    """Wraps embedding + vector store so retrieval code stays storage-agnostic.

    TODO:
      - Load the embedding model (ENV.embedding_model) via sentence-transformers.
      - Init a persistent Chroma collection at ENV.vector_store_dir.
      - build(): embed all chunks and upsert them with chunk_id as the id.
      - search(): embed the query, return top_k chunks with similarity scores.
    Keeping this swappable (FAISS/Qdrant) later is just re-implementing these two methods.
    """

    def __init__(self):
        self.model_name = ENV.embedding_model
        self.store_dir = ENV.vector_store_dir
        # TODO: init embedding model + chroma client

    def build(self, chunks: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[tuple[dict, float]]:
        raise NotImplementedError
