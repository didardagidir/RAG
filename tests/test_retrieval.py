"""Start your tests with the parts that have exact right answers: the metrics and
RRF. These don't need an LLM or a GPU, so they run fast and catch real bugs."""
from src.evaluation.retrieval_metrics import recall_at_k, mrr_at_k, ndcg_at_k


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0


def test_mrr_at_k():
    assert mrr_at_k(["a", "b", "c"], {"b"}, k=3) == 0.5
    assert mrr_at_k(["a", "b", "c"], {"a"}, k=3) == 1.0


def test_ndcg_perfect_is_one():
    # single relevant doc at rank 1 -> perfect ranking
    assert ndcg_at_k(["a", "b"], {"a"}, k=2) == 1.0
