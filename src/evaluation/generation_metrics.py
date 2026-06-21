"""Generation-side evaluation via RAGAS (faithfulness + answer relevance).

Faithfulness catches hallucination: is every claim in the answer supported by the
retrieved context? Answer relevance catches off-topic answers.
"""
from __future__ import annotations


def evaluate_generation(samples: list[dict]) -> dict:
    """samples = [{'question', 'answer', 'contexts', 'ground_truth'}].

    TODO:
      - Build a ragas EvaluationDataset from samples.
      - Run ragas.evaluate with [faithfulness, answer_relevancy].
      - Return averaged scores.
    Note: RAGAS itself calls an LLM, so this needs your API key set.
    """
    raise NotImplementedError
