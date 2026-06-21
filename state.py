"""CLI: ask a single question against the built index.

Usage: python scripts/run_query.py --question "Who founded the company that acquired DeepMind?"
"""
import argparse
# TODO: load the persisted BM25 + vector indexes, build_pipeline(...), pipe.ask(question),
# then print answer + citations + how many agent iterations it took.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True)
    args = ap.parse_args()
    raise NotImplementedError("Load indexes, build pipeline, ask, print result.")


if __name__ == "__main__":
    main()
