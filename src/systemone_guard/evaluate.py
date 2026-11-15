"""Frozen checkpoints only, test once after validation selection."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .baselines import probs as baseline_probs
from .causal import load_causal
from .causal import predict as causal_probs
from .data import CLASSES, load
from .metrics import score
from .train import load_checkpoint, predict_probs
from .types import Head, Pathish, Scores


def eval_split(
    split: str = "test",
    data: Pathish = "data",
    artifact: Pathish = "artifacts/guard_best/model.pt",
    out: Pathish = "experiments",
    include_causal: bool = True,
    artifacts: Pathish = "artifacts",
) -> dict[str, Scores]:
    rows = load(split, data)
    y = np.array([CLASSES.index(r["decision"]) for r in rows])
    m, t, temp = load_checkpoint(artifact)
    pred = predict_probs(m, rows, t, temperature=temp)
    names = {"systemone_guard": pred["decision"]}

    for name in ("word_tfidf", "char_tfidf"):
        file = Path(artifacts) / (name + ".pkl")

        if file.exists():
            with file.open("rb") as f:
                model = pickle.load(f)

            names[name] = baseline_probs(model, rows)

    if include_causal and (Path(artifacts) / "causal/model.pt").exists():
        c, ct = load_causal(Path(artifacts) / "causal/model.pt")
        names["causal_decoder_scratch"] = causal_probs(c, ct, rows)

    results = {name: score(p, y) for name, p in names.items()}

    auxiliary_heads: tuple[Head, ...] = ("sensitive", "approval_needed", "attack")
    aux: dict[str, float] = {
        k: float((pred[k].argmax(1) == np.array([r[k] for r in rows])).mean())
        for k in auxiliary_heads
    }
    results["systemone_guard"]["aux_accuracy"] = aux
    records = [
        {
            "id": row["id"],
            "gold": row["decision"],
            **{
                n: {
                    "prediction": CLASSES[int(probs[i].argmax())],
                    "probabilities": {
                        c: round(float(v), 6)
                        for c, v in zip(CLASSES, probs[i], strict=True)
                    },
                }
                for n, probs in names.items()
            },
        }
        for i, row in enumerate(rows)
    ]
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{split}_metrics.json").write_text(json.dumps(results, indent=2))

    with (output / f"{split}_predictions.jsonl").open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")

    print(
        split,
        json.dumps(
            {
                k: {
                    "accuracy": round(v["accuracy"], 4),
                    "macro_f1": round(v["macro_f1"], 4),
                    "ece": round(v["ece_10"], 4),
                }
                for k, v in results.items()
            },
            indent=2,
        ),
    )

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test", "challenge"], default="test")
    p.add_argument("--data", default="data")
    p.add_argument("--artifact", default="artifacts/guard_best/model.pt")
    p.add_argument("--out", default="experiments")
    p.add_argument("--artifacts", default="artifacts")
    a = p.parse_args()
    eval_split(**vars(a))


if __name__ == "__main__":
    main()
