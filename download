"""Split passages into retrieval-sized chunks.

Chunk size is a real lever on RAG quality: too big and retrieval is imprecise,
too small and you lose context. config.yaml controls it.
"""
from __future__ import annotations

from src.config import CONFIG


def chunk_passages(corpus: list[dict]) -> list[dict]:
    """Split each passage into overlapping chunks.

    TODO:
      - Use a token-aware splitter (e.g. langchain RecursiveCharacterTextSplitter
        with a tokenizer, or tiktoken) sized by CONFIG.data['chunk_size'] /
        ['chunk_overlap'].
      - Keep passage_id on each chunk so you can trace a chunk back to its source.
    Returns: list of {"chunk_id": str, "passage_id": str, "text": str}
    """
    raise NotImplementedError
