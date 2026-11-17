"""Regenerate published comparisons from the two models' matched raw records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

from ..types import Pathish
from .analysis import GUARD_METHOD, METHODS, SMOLLM_METHOD, compare_records
from .comparison import AUXILIARY_HEADS, CLASSES, summarize
from .types import BenchmarkResult, Summary

SPLITS = (
    ("test", "Test"),
    ("challenge", "Challenge"),
    ("manual_probes", "Manual probes"),
    ("generalization", "New wording"),
)
DISPLAY_NAMES = {GUARD_METHOD: "Guard", SMOLLM_METHOD: "SmolLM2 LoRA"}
START = "<!-- BEGIN MEASURED COMPARISON -->"
END = "<!-- END MEASURED COMPARISON -->"


def parameter_counts(result: BenchmarkResult) -> dict[str, int]:
    details = cast(
        dict[str, dict[str, object]], result["metadata"].get("model_details", {})
    )

    return {
        method: int(str(details.get(method, {}).get("parameters", 0)))
        for method in METHODS
    }


def load_reports(experiments: Pathish) -> dict[str, BenchmarkResult]:
    """Reject stale derived values, extra methods, or unmatched examples/timings."""
    reports = {}

    for split, _ in SPLITS:
        result = cast(
            BenchmarkResult,
            json.loads((Path(experiments) / f"finetuned_{split}.json").read_text()),
        )
        expected = compare_records(
            result["records"], int(str(result["metadata"]["seed"]))
        )

        if result["comparison"] != expected:
            raise ValueError(f"{split}: saved effects differ from the raw records")

        summaries = [summarize(result["records"], method) for method in METHODS]

        if result["summaries"] != summaries:
            raise ValueError(f"{split}: saved summaries differ from the raw records")

        reports[split] = result

    return reports


def summary_for(result: BenchmarkResult, method: str) -> Summary:
    for summary in result["summaries"]:
        if summary is not None and summary["method"] == method:
            return summary

    raise ValueError(f"Missing summary for {method}")


def overview(reports: dict[str, BenchmarkResult]) -> str:
    counts = parameter_counts(reports["test"])
    fresh = reports["generalization"]
    lines = [
        "Measured comparison with **SmolLM2-135M-Instruct fine-tuned using LoRA** "
        f"({counts[SMOLLM_METHOD]:,} merged parameters) and **SystemOne Guard** ({counts[GUARD_METHOD]:,} parameters). "
        "Both return the decision and all three auxiliary outputs.",
        "",
        "| Split | Guard correct / total | LoRA correct / total | Guard accuracy | LoRA accuracy | LoRA - Guard |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for split, label in SPLITS:
        effect = reports[split]["comparison"]
        n = effect["quality_n"]
        lines.append(
            f"| {label} | {effect['guard_correct']}/{n} | {effect['smollm_correct']}/{n} "
            f"| {100 * effect['guard_correct'] / n:.4f}% | {100 * effect['smollm_correct'] / n:.4f}% "
            f"| {effect['accuracy_gain_pp']:+.4f} pp ({effect['additional_correct']:+d} correct) |"
        )

    lines.extend(
        [
            "",
            "Accuracy differences above are **percentage points**, with Guard as the accuracy baseline. "
            "Positive differences favor LoRA; negative differences favor Guard. "
            "The previously inspected manual probes are a development benchmark. "
            f"The final wording set has {fresh['comparison']['quality_n']} examples from "
            f"{fresh['metadata']['unique_contexts']} separately authored contexts and "
            "was frozen before candidate training; it is still a synthetic diagnostic.",
            "",
            "| Split | Timed requests per model | Guard median | LoRA median | Guard saves | Guard latency reduction | Guard speedup |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for split, label in SPLITS:
        comparison = reports[split]["comparison"]
        effect_ms = comparison["latency"]["median"]
        lines.append(
            f"| {label} | {comparison['latency_n']} | {effect_ms['guard_ms']:.3f} ms "
            f"| {effect_ms['smollm_ms']:.3f} ms | {effect_ms['saved_ms']:.3f} ms "
            f"| {effect_ms['guard_reduction_percent']:.2f}% | {effect_ms['guard_speedup']:.2f}x |"
        )

    metadata = reports["test"]["metadata"]
    hardware = metadata["gpu_name"] or str(metadata["device"]).upper()
    lines.extend(
        [
            "",
            f"Latency is measured on the **same {hardware}**, FP32, batch size 1, "
            f"{metadata['cpu_threads']} host threads, with device synchronization and tokenization, all four outputs, "
            "decoding and probability extraction included. Both models remain loaded; "
            "measurement order alternates per example after five warmup examples each. "
            "Loading, adapter merging, and warmup are excluded. Quality uses every example; "
            "untimed records have null latency. Quality batches are not reported as request latency.",
            "",
            "**Latency reduction** = `100 x (LoRA ms - Guard ms) / LoRA ms`. "
            "**Speedup** = `LoRA ms / Guard ms`. Each ratio uses the same statistic "
            "(median, p95, or mean); reduction is not a speedup percentage.",
        ]
    )

    lines.extend(
        [
            "",
            "The recorded inference workers used a 1.5 GiB host-RAM cap and a 150% CPU quota "
            "to preserve desktop responsiveness. These are measurements of this constrained "
            "reference implementation. Previously executed LoRA quality may be reused only "
            "when model, input and prompt hashes match; every reported latency is freshly measured.",
        ]
    )

    return "\n".join(lines)


def detailed_report(reports: dict[str, BenchmarkResult], experiments: Path) -> str:
    training = json.loads((experiments / "smollm2_finetuning.json").read_text())
    guard_training = json.loads((experiments / "guard_training.json").read_text())
    fresh_contexts = reports["generalization"]["metadata"]["unique_contexts"]
    lines = [
        "# SystemOne Guard versus fine-tuned SmolLM2",
        "",
        overview(reports),
        "",
        "## Decision quality and calibration",
        "",
        "The main prediction is the highest validation-calibrated decision probability. "
        "SmolLM2 decodes four constrained tokens; its decision probabilities are normalized "
        "over allow/review/block, and its binary probabilities condition on preceding decoded slots. "
        "Guard predicts all four heads in one forward pass. NLL, multiclass Brier (sum over classes), "
        "and 10-bin ECE are lower-is-better. ECE is a fraction, not a percentage.",
        "",
        "| Split | Model | n | Accuracy | Macro-F1 | NLL | Brier | ECE-10 | Invalid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split, label in SPLITS:
        for method in METHODS:
            row = summary_for(reports[split], method)
            lines.append(
                f"| {label} | {DISPLAY_NAMES[method]} | {row['n']} | {100 * row['accuracy']:.4f}% "
                f"| {row['macro_f1']:.6f} | {row['nll']:.8g} | {row['brier']:.8g} "
                f"| {row['ece_10']:.8g} | {100 * row['invalid_rate']:.2f}% |"
            )

    test_effect = reports["test"]["comparison"]
    challenge_effect = reports["challenge"]["comparison"]
    probe_effect = reports["manual_probes"]["comparison"]
    lines.extend(
        [
            "",
            "![Accuracy and exact changes](../figures/comparison.png)",
            "",
            "![Decision calibration](../figures/calibration.png)",
            "",
            "## Accuracy effects and uncertainty",
            "",
            "Absolute gain uses `100 x (LoRA correct - Guard correct) / n`. Relative gain "
            "uses `100 x (LoRA correct - Guard correct) / Guard correct`; these are different quantities. "
            "Intervals use 10,000 paired bootstrap resamples of examples, seed 42. "
            "They describe these fixed datasets, not variation across training seeds or real deployments. "
            f"The final wording set reuses each of {fresh_contexts} context strings across four typed states; "
            "example-level intervals do not account for that context clustering.",
            "",
            "| Split | LoRA absolute gain | LoRA relative gain | Paired 95% interval (pp) |",
            "|---|---:|---:|---:|",
        ]
    )

    for split, label in SPLITS:
        effect = reports[split]["comparison"]
        low, high = effect["accuracy_gain_ci95_pp"]
        relative = effect["accuracy_relative_gain_percent"]
        relative_text = "undefined" if relative is None else f"{relative:+.4f}%"
        lines.append(
            f"| {label} | {effect['accuracy_gain_pp']:+.4f} pp | {relative_text} | [{low:+.4f}, {high:+.4f}] |"
        )

    lines.extend(
        [
            "",
            f"The synthetic test difference is {test_effect['additional_correct']:+d} correct examples "
            f"out of {test_effect['quality_n']}; the challenge difference is "
            f"{challenge_effect['additional_correct']:+d} out of {challenge_effect['quality_n']}. "
            f"Read the manual-probe difference with its {probe_effect['quality_n']}-example "
            "denominator and hand-selected composition. A paired interval containing zero "
            "does not establish a nonzero accuracy improvement.",
            "",
            "## Request latency",
            "",
            "p95 uses `numpy.quantile(times, 0.95, method='linear')`. "
            "All statistics use the same timed IDs for both models. "
            "A hundred requests provide a limited estimate of tail latency; thermal state, host load, "
            "hardware and library versions affect these measurements. These are measurements of "
            "this implementation and decoding protocol, not an optimized inference-runtime limit.",
            "",
            "| Split | Statistic | Guard ms | LoRA ms | Saved ms | Guard reduction | Guard speedup |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )

    for split, label in SPLITS:
        for statistic, effect_ms in reports[split]["comparison"]["latency"].items():
            lines.append(
                f"| {label} | {statistic} | {effect_ms['guard_ms']:.3f} | {effect_ms['smollm_ms']:.3f} "
                f"| {effect_ms['saved_ms']:.3f} | {effect_ms['guard_reduction_percent']:.2f}% "
                f"| {effect_ms['guard_speedup']:.2f}x |"
            )

    lines.extend(
        [
            "",
            "![Matched request latency](../figures/latency.png)",
            "",
            "## Auxiliary heads",
            "",
            "| Split | Model | Sensitive | Approval needed | Attack |",
            "|---|---|---:|---:|---:|",
        ]
    )

    for split, label in SPLITS:
        for method in METHODS:
            row = summary_for(reports[split], method)
            aux_values = " | ".join(
                f"{100 * row['aux_accuracy'][head]:.4f}%" for head in AUXILIARY_HEADS
            )
            lines.append(f"| {label} | {DISPLAY_NAMES[method]} | {aux_values} |")

    lines.extend(
        [
            "",
            "![Auxiliary accuracy](../figures/auxiliary_accuracy.png)",
            "",
            "## Per-class decision metrics",
            "",
            "| Split | Model | Class | Precision | Recall | F1 | Support |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )

    for split, label in SPLITS:
        for method in METHODS:
            for category in CLASSES:
                row_class = summary_for(reports[split], method)["per_class"][category]
                lines.append(
                    f"| {label} | {DISPLAY_NAMES[method]} | {category} | {row_class['precision']:.6f} "
                    f"| {row_class['recall']:.6f} | {row_class['f1-score']:.6f} | {int(row_class['support'])} |"
                )

    lines.extend(
        [
            "",
            "![Both models' confusion matrices](../figures/confusion.png)",
            "",
            "## Free generation check",
            "",
            "SmolLM2 is also evaluated with greedy free generation (up to eight tokens). "
            "Only a decision plus three yes/no words parses successfully; invalid output counts "
            "as wrong. This is a separate diagnostic from the constrained main comparison. "
            "Its latency is not used to calculate Guard's main speedup.",
            "",
            "| Split | n | LoRA free decision accuracy | Invalid format | Median ms | p95 ms |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for split, label in SPLITS:
        generation = summary_for(reports[split], SMOLLM_METHOD)["generation"]
        lines.append(
            f"| {label} | {generation['n']} | {100 * generation['accuracy']:.4f}% | {100 * generation['invalid_rate']:.2f}% | {generation['median_latency_ms']:.3f} | {generation['p95_latency_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Manual-probe groups and errors",
            "",
            "| Group | n | Guard correct | LoRA correct |",
            "|---|---:|---:|---:|",
        ]
    )
    guard_groups = summary_for(reports["manual_probes"], GUARD_METHOD)["by_group"]
    smollm_groups = summary_for(reports["manual_probes"], SMOLLM_METHOD)["by_group"]

    for group, values in guard_groups.items():
        n = int(values["n"])
        lines.append(
            f"| {group} | {n} | {round(n * values['accuracy'])}/{n} | {round(n * smollm_groups[group]['accuracy'])}/{n} |"
        )

    lines.extend(
        [
            "",
            "These probes were inspected during error analysis, so their post-refactor score is "
            "a development result. The separate new wording set provides an additional check "
            "on disjoint context strings. It does not establish real-world security robustness.",
            "",
            "| Split | Model | Incorrect example | Gold | Prediction |",
            "|---|---|---|---|---|",
        ]
    )

    for split, label in SPLITS:
        for record in reports[split]["records"]:
            if record["prediction"] != record["gold"]:
                lines.append(
                    f"| {label} | {DISPLAY_NAMES[record['method']]} | {record['id']} | {record['gold']} | {record['prediction']} |"
                )

    lines.extend(
        [
            "",
            "## Final wording groups",
            "",
            "| Group | n | Guard correct | LoRA correct |",
            "|---|---:|---:|---:|",
        ]
    )
    new_guard = summary_for(reports["generalization"], GUARD_METHOD)["by_group"]
    new_smollm = summary_for(reports["generalization"], SMOLLM_METHOD)["by_group"]

    for group, values in new_guard.items():
        n_group = int(values["n"])
        lines.append(
            f"| {group} | {n_group} | {round(n_group * values['accuracy'])}/{n_group} | {round(n_group * new_smollm[group]['accuracy'])}/{n_group} |"
        )

    lines.extend(
        [
            "",
            "Overall accuracy does not establish stronger attack detection. On this final set, Guard misses 24/72 untrusted attack cases, while LoRA detects all 72 but makes many errors on benign context. The predeclared 95% overall wording-set target is still unmet.",
        ]
    )

    metadata = reports["test"]["metadata"]
    lines.extend(
        [
            "",
            "## Training and comparison scope",
            "",
            f"Guard uses {guard_training['train_examples']:,} training examples and "
            f"{guard_training['validation_examples']:,} validation examples, including broader "
            "compositional attack/benign contexts. The frozen SmolLM2 LoRA adapter uses its "
            "original 10,000 training and 1,500 validation examples. Task data budgets now differ. "
            "Checkpoint selection and temperature fitting use validation only. "
            "Test, challenge, manual probes and the frozen new wording set are excluded from "
            "gradient updates and epoch selection. Manual-probe failures did motivate the redesign. "
            "This matches evaluation inputs and hardware, but not prior training, task data, architecture, "
            "parameter budget or total training compute. SmolLM2 also receives the policy in its prompt. "
            "The comparison measures quality and cost for these two implementations on this task.",
            "",
            f"LoRA uses rank {training['rank']}, alpha {training['lora_alpha']}, all-linear targets, "
            f"{training['trainable_parameters']:,} trainable parameters, three epochs, "
            "BF16, microbatch 8, accumulation 4, learning rate 0.0003, and seed 42. "
            f"Epoch {training['selected_epoch']} is selected and merged into FP32 for inference. "
            f"Training took {training['seconds'] / 60:.2f} minutes; peak allocation was "
            f"{training['peak_allocated_mib'] / 1024:.2f} GiB and peak reservation "
            f"{training['peak_reserved_mib'] / 1024:.2f} GiB. Guard trains four public heads "
            "plus a source-independent context-intent head from random initialization. LoRA uses response-token cross entropy. "
            "Their loss values and historical training runtimes are not directly comparable.",
            "",
            "![Both training traces](../figures/training.png)",
            "",
            f"Evaluation environment: {metadata['gpu_name']}; PyTorch {metadata['torch_version']}; "
            f"Transformers {metadata['transformers_version']}; CUDA {metadata['cuda_version']}. "
            "TF32 is disabled. Reports record model hashes, revision, temperatures, prompt, "
            "selected input IDs, timing IDs, and software versions.",
            "",
            "## Regenerate",
            "",
            "```bash",
            "make setup-gpu",
            "# Reuse the completed adapter, or create it once with: make finetune",
            "make improve-guard",
            "make benchmark ARGS='--split test'",
            "make benchmark ARGS='--split challenge'",
            "make benchmark ARGS='--split manual_probes'",
            "make benchmark ARGS='--split generalization'",
            "make plots",
            "make report",
            "```",
            "",
            "Sources: [test](../experiments/finetuned_test.json), "
            "[challenge](../experiments/finetuned_challenge.json), "
            "[probes](../experiments/finetuned_manual_probes.json), "
            "[new wording](../experiments/finetuned_generalization.json), "
            "[Guard training](../experiments/guard_training.json), "
            "[LoRA training](../experiments/smollm2_finetuning.json). "
            "Run `make report` after new measurements; this document and the overview tables are generated.",
        ]
    )

    return "\n".join(lines) + "\n"


def update_overview(path: Path, text: str) -> None:
    content = path.read_text()
    replacement = f"{START}\n\n{text}\n\n{END}"

    if START not in content or END not in content:
        raise ValueError(f"Missing measured-comparison markers in {path}")

    path.write_text(
        re.sub(
            re.escape(START) + r".*?" + re.escape(END),
            lambda _: replacement,
            content,
            flags=re.DOTALL,
        )
    )


def write_improvement(reports: dict[str, BenchmarkResult], root: Path) -> None:
    """Keep the original failure visible and assess predeclared experiment targets."""
    path = root / "experiments/guard_improvement.json"
    audit = json.loads(path.read_text())
    manual = reports["manual_probes"]["comparison"]
    fresh = reports["generalization"]["comparison"]
    targets = {
        "manual_at_least_frozen_smollm": manual["guard_correct"]
        >= manual["smollm_correct"],
        "test_accuracy_at_least_99_5_percent": summary_for(
            reports["test"], GUARD_METHOD
        )["accuracy"]
        >= 0.995,
        "challenge_accuracy_at_least_99_5_percent": summary_for(
            reports["challenge"], GUARD_METHOD
        )["accuracy"]
        >= 0.995,
        "new_wording_accuracy_at_least_95_percent": summary_for(
            reports["generalization"], GUARD_METHOD
        )["accuracy"]
        >= 0.95,
        "median_latency_below_one_tenth_of_smollm": all(
            report["comparison"]["latency"]["median"]["guard_speedup"] >= 10
            for report in reports.values()
        ),
    }
    audit.update(
        status="evaluated",
        targets=targets,
        all_targets_met=all(targets.values()),
        selected_guard_parameters=parameter_counts(reports["test"])[GUARD_METHOD],
        current_comparisons={
            split: report["comparison"] for split, report in reports.items()
        },
    )
    path.write_text(json.dumps(audit, indent=2) + "\n")
    lines = [
        "## Measured outcome",
        "",
        "| Set | Original guard correct | Updated guard correct | Frozen SmolLM2 LoRA correct |",
        "|---|---:|---:|---:|",
    ]

    for split, label in SPLITS:
        comparison = reports[split]["comparison"]
        n = comparison["quality_n"]
        baseline = audit["baseline_comparison_summaries"].get(split, [])
        original = next(
            (row for row in baseline if row["method"] == GUARD_METHOD), None
        )
        old_text = (
            f"{round(original['n'] * original['accuracy'])}/{original['n']}"
            if original
            else "not measured"
        )
        lines.append(
            f"| {label} | {old_text} | {comparison['guard_correct']}/{n} | {comparison['smollm_correct']}/{n} |"
        )

    lines.extend(["", "| Predeclared target | Result |", "|---|---|"])
    lines.extend(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in targets.items()
    )
    lines.extend(
        [
            "",
            f"Updated Guard has {audit['selected_guard_parameters']:,} parameters. "
            "The original 21,306-parameter checkpoint remains loadable and unchanged. "
            "The updated model receives broader task training data; the SmolLM2 adapter remains frozen.",
            "",
            f"The fresh wording check is {fresh['guard_correct']}/{fresh['quality_n']} for Guard and "
            f"{fresh['smollm_correct']}/{fresh['quality_n']} for LoRA. Its separately authored contexts are "
            "repeated across four typed states and do not establish general language-model equivalence.",
            "",
            "See [the complete comparison](FINETUNED_COMPARISON.md) for quality, calibration, "
            "auxiliary outputs, incorrect IDs, matched latency, and data-budget qualifications.",
        ]
    )
    document = root / "docs/GUARD_IMPROVEMENT.md"
    content = (
        document.read_text().split("<!-- BEGIN IMPROVEMENT RESULTS -->")[0].rstrip()
    )
    document.write_text(
        content
        + "\n\n<!-- BEGIN IMPROVEMENT RESULTS -->\n\n"
        + "\n".join(lines)
        + "\n\n<!-- END IMPROVEMENT RESULTS -->\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    reports = load_reports(args.root / "experiments")
    (args.root / "docs/FINETUNED_COMPARISON.md").write_text(
        detailed_report(reports, args.root / "experiments")
    )

    for path in (args.root / "README.md", args.root / "hf_model/README.md"):
        update_overview(path, overview(reports))

    write_improvement(reports, args.root)


if __name__ == "__main__":
    main()
