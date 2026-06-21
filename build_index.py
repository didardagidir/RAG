"""CLI: load a dataset, build the corpus, chunk it, and build both indexes.
Run this once before querying.

Usage: python scripts/build_index.py --dataset natural_questions --sample 5000
"""
import argparse
from src.data.loaders import load_dataset_normalized
from src.data.preprocess import build_corpus
from src.indexing.chunking import chunk_passages
from src.indexing.vector_index import VectorIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="natural_questions")
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    records = load_dataset_normalized(args.dataset, args.sample)
    corpus = build_corpus(records)
    chunks = chunk_passages(corpus)

    vidx = VectorIndex()
    vidx.build(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(records)} records.")
    # TODO: also persist the BM25 index (pickle chunks) so run_query can reload it.


if __name__ == "__main__":
    main()
