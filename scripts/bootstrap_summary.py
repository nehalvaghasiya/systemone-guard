"""Paired test bootstrap and dataset distribution analysis (no model tuning)."""

import argparse
import json
from pathlib import Path

import numpy as np

from systemone_guard.data import CLASSES, load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--input", default="experiments/test_predictions.jsonl")
    parser.add_argument("--out", default="experiments/bootstrap_summary.json")
    args = parser.parse_args()
    np.random.seed(7)
    rows = load("test", args.data)
    n = len(rows)
    recs = [json.loads(s) for s in Path(args.input).read_text().splitlines()]
    pair = {
        name: np.array([r[name]["prediction"] == r["gold"] for r in recs], dtype=float)
        for name in (
            "systemone_guard",
            "word_tfidf",
            "char_tfidf",
            "causal_decoder_scratch",
        )
    }

    comparisons: dict[str, dict[str, float | list[float]]] = {}
    res = {
        "sample_size": n,
        "method": "paired bootstrap resampling of test examples; 3000 replicates",
        "comparisons": comparisons,
    }
    idx = np.random.default_rng(7).integers(0, n, size=(3000, n))

    for other in ("word_tfidf", "char_tfidf", "causal_decoder_scratch"):
        delta = pair["systemone_guard"] - pair[other]
        samps = delta[idx].mean(1)
        comparisons[f"guard_minus_{other}"] = {
            "difference_pp": float(delta.mean() * 100),
            "ci95_pp": [float(x * 100) for x in np.quantile(samps, [0.025, 0.975])],
        }

    res["test_class_balance"] = {
        c: sum(r["decision"] == c for r in rows) for c in CLASSES
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
