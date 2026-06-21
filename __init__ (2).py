"""Dataset loaders — BEIR format (corpus / queries / qrels).

Why BEIR format instead of raw natural_questions:
  - Raw NQ is ~40GB with byte-offset annotations that are painful to parse.
  - BEIR ships NQ already as (corpus, queries, qrels), so gold relevant passage
    ids come for free — retrieval metrics work immediately.
  - It IS the BEIR benchmark, so Phase 3/5 evaluation reuses the same data.

Returned structures (used everywhere downstream):
  corpus  -> dict[passage_id, {"title": str, "text": str}]
  queries -> dict[query_id, question_str]
  qrels   -> dict[query_id, dict[passage_id, relevance_int]]
"""
from __future__ import annotations

import random

# BEIR public mirrors. nq for single-hop, hotpotqa for multi-hop (Phase 5).
BEIR_URLS = {
    "nq": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip",
    "hotpotqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip",
}


def load_beir_raw(name: str = "nq", split: str = "test", out_dir: str = "datasets"):
    """Download + load a BEIR dataset using the official `beir` loader.

    Returns the full (corpus, queries, qrels). The corpus here can be huge
    (NQ ~2.68M docs) — we shrink it in build_working_set below.
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = util.download_and_unzip(BEIR_URLS[name], out_dir)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    return corpus, queries, qrels


def build_working_set(
    corpus: dict,
    queries: dict,
    qrels: dict,
    n_queries: int = 200,
    n_distractors: int = 5000,
    seed: int = 42,
):
    """Shrink the data to something you can index on Colab — WITHOUT breaking
    evaluation.

    THE TRAP: if you just randomly sample the corpus, you'll probably throw away
    the gold passages your queries need, and every retrieval score becomes 0.
    So the rule is: keep every gold passage, then add random distractors.

    Steps:
      1. Pick n_queries queries (that actually have qrels).
      2. Collect all gold passage ids those queries point to -> ALWAYS keep these.
      3. Add n_distractors random non-gold passages so retrieval isn't trivially easy.
    """
    rng = random.Random(seed)

    # 1. queries that have at least one relevance judgment
    answerable = [qid for qid in queries if qrels.get(qid)]
    chosen_qids = rng.sample(answerable, min(n_queries, len(answerable)))
    sub_queries = {qid: queries[qid] for qid in chosen_qids}
    sub_qrels = {qid: qrels[qid] for qid in chosen_qids}

    # 2. gold passages we must keep
    gold_ids = {pid for qid in chosen_qids for pid in sub_qrels[qid]}

    # 3. distractors: random passages that are not gold
    non_gold = [pid for pid in corpus if pid not in gold_ids]
    distractor_ids = set(rng.sample(non_gold, min(n_distractors, len(non_gold))))

    keep_ids = gold_ids | distractor_ids
    sub_corpus = {pid: corpus[pid] for pid in keep_ids}

    return sub_corpus, sub_queries, sub_qrels


def corpus_to_passages(corpus: dict) -> list[dict]:
    """Flatten the corpus dict into a list the indexer can iterate.
    Returns: [{"passage_id", "title", "text"}]"""
    return [
        {"passage_id": pid, "title": doc.get("title", ""), "text": doc["text"]}
        for pid, doc in corpus.items()
    ]
