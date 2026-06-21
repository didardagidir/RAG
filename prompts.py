"""CLI: run the full evaluation harness and print a results table.

Usage: python scripts/evaluate.py --dataset hotpot_qa --n 200
"""
import argparse
# TODO: load dataset + indexes, run evaluate_retrieval and evaluate_generation,
# print a markdown table you can paste straight into the README Results section.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpot_qa")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
