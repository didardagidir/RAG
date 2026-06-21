# Learning guide — how to actually build this

This repo is a scaffold. The `NotImplementedError`s are deliberate: each one is a
thing worth understanding rather than copy-pasting. Suggested order and what to learn
at each step.

## Phase 1 — Data & indexing
- **Learn:** how HuggingFace `datasets` streaming works; tokenization-aware chunking;
  why chunk size matters for retrieval precision.
- **Files:** `src/data/loaders.py`, `src/data/preprocess.py`, `src/indexing/chunking.py`.
- **Done when:** you can print 3 normalized records and the corpus passage count.

## Phase 2 — Baseline RAG
- **Learn:** dense embeddings, cosine similarity, vector stores; why a baseline matters.
- **Files:** `src/indexing/vector_index.py`, `src/generation/generator.py`
  (`simple_generate`).
- **Done when:** `simple_generate` answers a question from retrieved passages.

## Phase 3 — Hybrid + rerank
- **Learn:** BM25 vs dense (lexical vs semantic), Reciprocal Rank Fusion, cross-encoders.
- **Files:** `src/retrieval/hybrid_retriever.py` (RRF is already written — study it),
  `src/retrieval/reranker.py`.
- **Done when:** you can show Recall@10 for BM25 vs dense vs hybrid. Record the numbers.

## Phase 4 — Agentic layer
- **Learn:** LangGraph state machines, conditional edges, the retrieve→grade→rewrite loop.
  This is the conceptual core. Read `src/agent/graph.py` carefully first — it already
  shows the shape; you implement the nodes.
- **Files:** `src/agent/nodes.py`.
- **Done when:** a multi-hop question produces sub-questions and loops more than once.

## Phase 5 — Multi-hop evaluation
- **Learn:** why single-retrieve fails on multi-hop; supporting-fact evaluation.
- **Files:** `src/evaluation/run_eval.py`.
- **Done when:** you have HotpotQA retrieval scores for non-agentic vs agentic.

## Phase 6 — Generation evaluation
- **Learn:** faithfulness/hallucination, LLM-as-judge, RAGAS.
- **Files:** `src/evaluation/generation_metrics.py`.
- **Done when:** the README Results table is filled in with real numbers.

## Phase 7 — Demo + writeup
- **Files:** `app/streamlit_app.py`.
- **Done when:** the app shows answer + citations + the agent's reasoning trace.

## A note on what makes this portfolio-worthy
The numbers and the *story of the numbers* are the differentiator. "Hybrid lifted
Recall@10 from X to Y, and the agentic loop lifted multi-hop accuracy from A to B"
is far more convincing than "I built a RAG app." Keep a short experiment log.
