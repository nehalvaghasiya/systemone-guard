"""Plot saved evaluation results or matched pretrained comparison results."""

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np

from systemone_guard.benchmarking.types import BenchmarkResult
from systemone_guard.data import CLASSES
from systemone_guard.types import Pathish


def create_results(
    experiments: Pathish = "experiments",
    artifacts: Pathish = "artifacts",
    output_dir: Pathish = "figures",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        name: json.loads(Path(experiments, name + "_metrics.json").read_text())
        for name in ("val", "test", "challenge")
    }
    models = list(metrics["test"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bar = 0.22

    for i, split in enumerate(["val", "test", "challenge"]):
        vals = [metrics[split][model]["accuracy"] * 100 for model in models]
        ax.bar(x + (i - 1) * bar, vals, width=bar, label=split)

    ax.set_xticks(x, [m.replace("_", "\n") for m in models])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Measured results across dataset splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=165)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array(metrics["test"]["systemone_guard"]["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("SystemOne Guard · Test confusion matrix")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=165)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, _color in [("run_d32", "tab:blue"), ("run_d48", "tab:orange")]:
        p = Path(artifacts) / name / "training.json"

        if p.exists():
            dat = json.loads(p.read_text())["history"]
            ax.plot(
                [d["epoch"] for d in dat],
                [d["train_weighted_loss"] for d in dat],
                marker="o",
                label=name,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted training loss")
    ax.set_yscale("log")
    ax.set_title("Training curves · measured")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "train_loss.png", dpi=165)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))

    for name in ["run_d32", "run_d48"]:
        p = Path(artifacts) / name / "training.json"

        if p.exists():
            dat = json.loads(p.read_text())["history"]
            ax.plot(
                [d["epoch"] for d in dat],
                [d["val_macro_f1"] for d in dat],
                marker="o",
                label=name,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0, 1.03)
    ax.set_title("Validation macro-F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "val_f1.png", dpi=165)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = np.arange(len(models))
    v = [metrics["test"][m]["ece_10"] for m in models]
    ax.bar(xs, v)
    ax.set_xticks(xs, [m.replace("_", "\n") for m in models])
    ax.set_ylabel("ECE (10 bins; lower better)")
    ax.set_title("Probability calibration · synthetic test")
    fig.tight_layout()
    fig.savefig(out / "calibration.png", dpi=165)
    plt.close(fig)
    print("Generated", sorted(str(p) for p in out.glob("*.png")))

    fig, ax = plt.subplots(figsize=(8, 4))

    for name in ["run_d16", "run_d32", "run_d48"]:
        p = Path(artifacts) / name / "training.json"

        if p.exists():
            dat = json.loads(p.read_text())["history"]
            ax.plot(
                [d["epoch"] for d in dat],
                [d["val_nll"] for d in dat],
                marker="o",
                label=name,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation negative log likelihood")
    ax.set_yscale("log")
    ax.set_title("Validation NLL (separate from training multi-task loss)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "val_loss.png", dpi=165)
    plt.close(fig)


def create_pretrained(input_path: Pathish, output_dir: Pathish) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = cast(BenchmarkResult, json.loads(Path(input_path).read_text()))
    summaries = [summary for summary in result["summaries"] if summary is not None]

    if len(summaries) < 2:
        raise ValueError("Need Guard and at least one pretrained model")

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    labels = [s["method"].replace("SmolLM2-", "SmolLM2\n") for s in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, [s["accuracy"] * 100 for s in summaries])
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(labels, [s["median_latency_ms"] for s in summaries])
    axes[1].set_ylabel("Median end-to-end latency (ms)")
    fig.suptitle(
        f"Pretrained LLM comparison · {result['metadata']['split']} · CPU · n={result['metadata']['n']}"
    )
    fig.tight_layout()
    fig.savefig(dest / "accuracy_latency.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))

    for s in summaries:
        ax.scatter(s["median_latency_ms"], 100 * s["accuracy"], label=s["method"], s=85)

    ax.set_xlabel("Median latency, single request (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "quality_vs_latency.png", dpi=150)
    plt.close(fig)
    print(
        "Created", dest / "accuracy_latency.png", "and", dest / "quality_vs_latency.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    results = commands.add_parser(
        "results", help="Plot saved split metrics and training histories"
    )
    results.add_argument("--experiments", default="experiments")
    results.add_argument("--artifacts", default="artifacts")
    results.add_argument("--out", default="figures")
    pretrained = commands.add_parser(
        "pretrained", help="Plot a matched pretrained benchmark"
    )
    pretrained.add_argument("--input", default="experiments/pretrained_comparison.json")
    pretrained.add_argument("--out", default="figures/pretrained_comparison")
    args = parser.parse_args()

    if args.command == "results":
        create_results(args.experiments, args.artifacts, args.out)
    else:
        create_pretrained(args.input, args.out)


if __name__ == "__main__":
    main()
