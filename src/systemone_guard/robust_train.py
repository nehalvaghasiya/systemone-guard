"""Train a subword guard on broader contexts; select using validation alone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .augmentation import VERSION, augment, generalization_rows, rows_hash
from .checkpoints import load_checkpoint, read_checkpoint
from .data import CLASSES, load
from .metrics import score
from .train import fit_temperature, predict_probs, train


def prepare_data(
    data: Path, output: Path, seed: int, train_count: int = 80000, val_count: int = 2000
) -> dict[str, object]:
    original_train = load("train", data)
    original_val = load("val", data)
    train_rows = augment(original_train, train_count, seed)
    val_rows = augment(original_val, val_count, seed + 1, validation=True)
    holdout = generalization_rows()
    train_new = train_rows[len(original_train) :]
    val_new = val_rows[len(original_val) :]
    contexts = [
        set(row["context"] for row in rows) for rows in (train_new, val_new, holdout)
    ]

    if any(contexts[i] & contexts[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("New train/validation/holdout contexts must be disjoint")

    from .benchmarking.probes import build_probes

    probe_texts = {probe["context"] for probe in build_probes()}

    if any(probe_texts & bank for bank in contexts):
        raise ValueError("A manual probe leaked into the new wording banks")

    output.mkdir(parents=True, exist_ok=True)

    for split, rows in (("train", train_rows), ("val", val_rows)):
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    manifest: dict[str, object] = {
        "augmentation_version": VERSION,
        "train_examples": len(train_rows),
        "validation_examples": len(val_rows),
        "train_sha256": rows_hash(train_rows),
        "validation_sha256": rows_hash(val_rows),
        "generalization_examples": len(holdout),
        "generalization_sha256": rows_hash(holdout),
        "new_unique_contexts": {
            name: len(bank)
            for name, bank in zip(
                ("train", "validation", "holdout"), contexts, strict=True
            )
        },
        "selection": "validation macro-F1, with lower validation NLL breaking ties",
        "manual_probe_role": "previously inspected development benchmark; never training or selection",
        "holdout_role": "separate wording bank frozen before training; never training or selection",
        "comparison_scope": "broader task data for Guard; existing SmolLM2 LoRA adapter remains frozen",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return manifest


def finalize_checkpoint(
    directory: Path, log: Path, report_path: Path
) -> dict[str, object]:
    """Recover the validation-selected checkpoint after a resource-limited stop."""
    torch.set_num_threads(1)
    prior = json.loads((directory / "training.json").read_text())
    text = log.read_text()
    manifest, _ = json.JSONDecoder().raw_decode(text[text.index("{") :])
    stage = [
        json.loads(line) for line in text.splitlines() if line.startswith('{"epoch"')
    ]

    if not stage:
        raise ValueError("Recovery log has no completed validation epochs")

    selected = max(stage, key=lambda row: (row["val_macro_f1"], -row["val_nll"]))
    model, tokenizer, _ = load_checkpoint(directory / "model.pt")
    rows = load("val", directory / "data")
    y = np.array([CLASSES.index(row["decision"]) for row in rows])
    raw = score(predict_probs(model, rows, tokenizer, batch=16)["decision"], y)

    if abs(raw["macro_f1"] - selected["val_macro_f1"]) > 1e-6:
        raise ValueError(
            "Saved checkpoint does not match the selected validation epoch"
        )

    temperature = fit_temperature(model, rows, tokenizer, batch=16)
    probabilities = predict_probs(
        model, rows, tokenizer, batch=16, temperature=temperature
    )
    validation = score(probabilities["decision"], y)
    validation["aux_accuracy"] = {
        head: float(
            (
                probabilities[head].argmax(1) == np.array([row[head] for row in rows])
            ).mean()
        )
        for head in ("sensitive", "approval_needed", "attack")
    }
    checkpoint = read_checkpoint(directory / "model.pt")
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, directory / "model.tmp")
    (directory / "model.tmp").replace(directory / "model.pt")
    previous = (
        prior["history"]
        if prior.get("status") == "interrupted"
        else prior.get("initial_training_history", [])
    )
    result = {
        **prior,
        **manifest,
        "status": "calibrated_after_resource_stop",
        "training_status": "resource-stopped; best validation checkpoint retained",
        "initial_training_history": previous,
        "continuation_history": stage,
        "history": [
            *previous,
            *[{**row, "epoch": len(previous) + row["epoch"]} for row in stage],
        ],
        "selected_epoch": len(previous) + selected["epoch"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "temperature": temperature,
        "selected_fp32_validation": raw,
        "calibrated_validation": validation,
        "checkpoint_sha256": hashlib.sha256(
            (directory / "model.pt").read_bytes()
        ).hexdigest(),
        "runtime_seconds": None,
        "runtime_note": "Training stopped before complete timing/peak statistics were captured",
    }
    (directory / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/guard_scaled"))
    parser.add_argument(
        "--report", type=Path, default=Path("experiments/guard_training.json")
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--finalize-log",
        type=Path,
        help="Calibrate the saved best checkpoint on CPU after a resource stop, using its captured run log",
    )
    args = parser.parse_args()

    if args.finalize_log is not None:
        finalize_checkpoint(args.out, args.finalize_log, args.report)

        return

    training_data = args.out / "data"
    manifest = prepare_data(args.data, training_data, args.seed)
    print(json.dumps(manifest, indent=2), flush=True)
    result = train(
        training_data,
        args.out,
        seed=args.seed,
        epochs=args.epochs,
        dim=64,
        hidden=128,
        lr=args.lr,
        batch=32,
        patience=8,
        device=args.device,
        subword_buckets=4096,
        max_len=160,
        max_words=4096,
        attack_weight=1.0,
        nll_tiebreak=True,
        mean_pool=True,
        attention_layers=3,
        context_only=True,
        factorized_policy=True,
        mixed_precision=args.device == "cuda",
        stream_features=True,
        accumulation=4,
        validation_batch=32,
        initial_checkpoint=args.resume_from,
        gpu_memory_fraction=0.2 if args.device == "cuda" else None,
    )
    result.update(manifest)
    (args.out / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
