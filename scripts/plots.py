"""Generate the six canonical Guard-versus-fine-tuned-SmolLM2 figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from systemone_guard.benchmarking.analysis import METHODS
from systemone_guard.benchmarking.reporting import (
    DISPLAY_NAMES,
    SPLITS,
    load_reports,
    parameter_counts,
    summary_for,
)
from systemone_guard.types import Pathish

COLORS = ("#2463a4", "#de8324")
NAMES = ("Guard", "SmolLM2 LoRA")


def _save(figure: Figure, output: Path, name: str) -> None:
    figure.savefig(output / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _bars(axis: Axes, values: list[float], suffix: str = "%") -> None:
    bars = axis.bar(
        NAMES,
        values,
        color=COLORS,
        width=0.55,
    )

    for bar, value in zip(bars, values, strict=True):
        axis.annotate(
            f"{value:.4f}{suffix}" if suffix == "%" else f"{value:.3f}{suffix}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def create_plots(
    experiments: Pathish = "experiments", out: Pathish = "figures"
) -> None:
    experiments, output = Path(experiments), Path(out)
    reports = load_reports(experiments)
    columns = len(SPLITS)
    counts = parameter_counts(reports["test"])
    names = (f"Guard · {counts[METHODS[0]]:,} params", "SmolLM2 LoRA · 135M")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.titlesize": 15})

    figure = plt.figure(figsize=(4.4 * columns, 5.7), layout="constrained")
    grid = figure.add_gridspec(2, columns, height_ratios=[3.3, 1.15])

    for index, (split, label) in enumerate(SPLITS):
        result = reports[split]
        comparison = result["comparison"]
        axis = figure.add_subplot(grid[0, index])
        _bars(
            axis, [100 * summary_for(result, method)["accuracy"] for method in METHODS]
        )
        axis.set_ylim(0, 110)
        axis.set_ylabel("Decision accuracy (%)")
        axis.set_title(f"{label} · n={comparison['quality_n']:,}")
        card = figure.add_subplot(grid[1, index])
        card.set_axis_off()
        card.text(
            0.5,
            0.8,
            f"LoRA - Guard: {comparison['accuracy_gain_pp']:+.4f} pp",
            ha="center",
            fontsize=13,
            weight="bold",
        )
        card.text(
            0.5,
            0.43,
            f"{comparison['additional_correct']:+d} correct decisions / {comparison['quality_n']:,}",
            ha="center",
        )
        card.text(
            0.5,
            0.08,
            f"Guard {comparison['guard_correct']:,} correct · LoRA {comparison['smollm_correct']:,} correct",
            ha="center",
            fontsize=9,
        )

    figure.suptitle(
        "Decision quality: Guard versus task-fine-tuned SmolLM2\nPercentage-point changes use all examples, not just the timed subset"
    )
    _save(figure, output, "comparison.png")

    figure, axes = plt.subplots(
        1, columns, figsize=(4.6 * columns, 5), layout="constrained"
    )

    for axis, (split, label) in zip(axes, SPLITS, strict=True):
        comparison = reports[split]["comparison"]
        x = np.arange(2)

        for index, method in enumerate(METHODS):
            summary = summary_for(reports[split], method)
            values = [summary["median_latency_ms"], summary["p95_latency_ms"]]
            bars = axis.bar(
                x + (index - 0.5) * 0.35,
                values,
                width=0.35,
                color=COLORS[index],
                label=names[index],
            )

            for bar, value in zip(bars, values, strict=True):
                axis.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )

        effect = comparison["latency"]["median"]
        axis.set_title(f"{label} · {comparison['latency_n']} matched requests")
        axis.set_xticks(x, ["Median", "p95"])
        axis.set_yscale("log")
        axis.set_ylim(
            0.5,
            max(
                comparison["latency"]["p95"]["smollm_ms"],
                comparison["latency"]["p95"]["guard_ms"],
            )
            * 2.5,
        )
        axis.set_ylabel("Request latency (ms, log scale)")
        axis.grid(axis="y", which="both", alpha=0.15)
        axis.set_axisbelow(True)
        axis.text(
            0.5,
            -0.16,
            f"Guard median: {effect['guard_reduction_percent']:.2f}% less latency\n{effect['guard_speedup']:.2f}x speedup · saves {effect['saved_ms']:.2f} ms",
            transform=axis.transAxes,
            ha="center",
            fontsize=10,
        )

    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Same GPU · FP32 · batch 1 · all four outputs · lower is better\nReduction = 100 x (LoRA - Guard) / LoRA; speedup = LoRA / Guard"
    )
    _save(figure, output, "latency.png")

    figure, axes = plt.subplots(
        3, columns, figsize=(4.4 * columns, 10), layout="constrained"
    )

    for column, (split, label) in enumerate(SPLITS):
        summaries = [summary_for(reports[split], method) for method in METHODS]
        scores = [
            (summary["nll"], summary["brier"], summary["ece_10"])
            for summary in summaries
        ]

        for row, title in enumerate(
            ("Negative log-likelihood", "Multiclass Brier", "ECE · 10 bins")
        ):
            axis = axes[row, column]
            values = [metrics[row] for metrics in scores]
            bars = axis.bar(
                NAMES, [max(value, 1e-9) for value in values], color=COLORS, width=0.55
            )
            axis.set_yscale("log")
            axis.set_ylim(
                max(min(value for value in values if value > 0) / 5, 1e-9)
                if any(values)
                else 1e-9,
                max(max(values) * 5, 1e-7),
            )
            axis.set_title(f"{label} · {title}")
            axis.set_ylabel("Score (log scale)")

            for bar, value in zip(bars, values, strict=True):
                axis.annotate(
                    f"{value:.5g}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )

    figure.suptitle(
        "Decision calibration · both models calibrated on validation only\nAll quality examples · lower is better · ECE shown as a fraction"
    )
    _save(figure, output, "calibration.png")

    figure, axes = plt.subplots(
        2, columns, figsize=(4.4 * columns, 8), layout="constrained"
    )

    for column, (split, label) in enumerate(SPLITS):
        for row, method in enumerate(METHODS):
            axis = axes[row, column]
            matrix = np.asarray(
                summary_for(reports[split], method)["confusion_matrix"]
            )[:3, :4]
            row_sums = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix, dtype=float),
                where=row_sums != 0,
            )
            axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            axis.set_xticks(range(4), ["allow", "review", "block", "invalid"])
            axis.set_yticks(range(3), ["allow", "review", "block"])
            axis.set_xlabel("Predicted decision")
            axis.set_ylabel("Gold decision")
            axis.set_title(f"{label} · {DISPLAY_NAMES[method]}")

            for (y, x), count in np.ndenumerate(matrix):
                axis.text(
                    x,
                    y,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if normalized[y, x] > 0.55 else "black",
                )

    figure.suptitle(
        "Decision confusion · labels show raw example counts\nColor shows the fraction within each gold-label row"
    )
    _save(figure, output, "confusion.png")

    figure, axes = plt.subplots(
        1, columns, figsize=(4.6 * columns, 4.8), layout="constrained"
    )

    for axis, (split, label) in zip(axes, SPLITS, strict=True):
        x = np.arange(3)

        for index, method in enumerate(METHODS):
            values = [
                100 * summary_for(reports[split], method)["aux_accuracy"][head]
                for head in ("sensitive", "approval_needed", "attack")
            ]
            bars = axis.bar(
                x + (index - 0.5) * 0.36,
                values,
                width=0.36,
                color=COLORS[index],
                label=names[index],
            )

            for bar, value in zip(bars, values, strict=True):
                axis.annotate(
                    f"{value:.2f}",
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    rotation=90,
                )

        axis.set_title(f"{label} · n={reports[split]['comparison']['quality_n']:,}")
        axis.set_xticks(x, ["Sensitive", "Approval", "Attack"])
        axis.set_ylabel("Auxiliary accuracy (%)")
        axis.set_ylim(0, 120)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)

    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("All three auxiliary outputs · Guard versus fine-tuned SmolLM2")
    _save(figure, output, "auxiliary_accuracy.png")

    guard = json.loads((experiments / "guard_training.json").read_text())
    smollm = json.loads((experiments / "smollm2_finetuning.json").read_text())
    guard_history, smollm_history = guard["history"], smollm["history"]
    guard_epochs = [row["epoch"] for row in guard_history]
    smollm_epochs = [row["epoch"] for row in smollm_history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes[0, 0].plot(
        guard_epochs,
        [row["train_weighted_loss"] for row in guard_history],
        "o-",
        color=COLORS[0],
    )
    axes[0, 0].set_title("Guard training · weighted four-head loss")
    axes[0, 1].plot(
        smollm_epochs,
        [row["training_loss"] for row in smollm_history],
        "o-",
        color=COLORS[1],
    )
    axes[0, 1].set_title("SmolLM2 LoRA training · response-token loss")

    for axis, metric, title in (
        (axes[1, 0], "macro_f1", "Validation macro-F1"),
        (axes[1, 1], "nll", "Validation NLL · before temperature fitting"),
    ):
        axis.plot(
            guard_epochs,
            [row[f"val_{metric}"] for row in guard_history],
            "o-",
            color=COLORS[0],
            label=names[0],
        )
        axis.plot(
            smollm_epochs,
            [row["validation"][metric] for row in smollm_history],
            "o-",
            color=COLORS[1],
            label=names[1],
        )
        axis.set_title(title)
        axis.legend(fontsize=9)

    validation_f1 = [row["val_macro_f1"] for row in guard_history] + [
        row["validation"]["macro_f1"] for row in smollm_history
    ]
    axes[1, 0].set_ylim(max(0, min(validation_f1) - 0.02), 1.005)
    axes[1, 1].set_yscale("log")

    for axis in axes.flat:
        axis.set_xlabel("Epoch (different model objectives and compute)")
        axis.grid(alpha=0.2)

    figure.suptitle(
        "Recorded training traces for both comparison models\nTraining losses have different definitions; they are shown separately"
    )
    _save(figure, output, "training.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    create_plots(args.experiments, args.out)


if __name__ == "__main__":
    main()
