"""Build the searchable corpus from normalized records.

The corpus is the flat pool of passages your retrievers search over. Keep a stable
passage_id so retrieval results can be matched back to gold labels during evaluation.
"""
from __future__ import annotations


def build_corpus(records: list[dict]) -> list[dict]:
    """Flatten all contexts into a deduplicated passage pool.

    Returns: list of {"passage_id": str, "text": str, "source_id": str}

    TODO:
      - Iterate records, pull out each context passage.
      - Deduplicate (the same Wikipedia passage appears across many questions).
      - Assign stable passage_ids.
    """
    raise NotImplementedError
