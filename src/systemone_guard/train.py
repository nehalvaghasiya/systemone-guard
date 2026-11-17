"""Training and validation-only model selection for parallel typed decisions."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .checkpoints import load_checkpoint as load_checkpoint
from .checkpoints import read_checkpoint
from .data import CLASSES, load
from .metrics import score
from .model import LABELS, GuardNet, Tokenizer, tensors
from .training_cache import cache_features, check_host_memory
from .types import FloatArray, Head, LabeledInput, Pathish


def predict_probs(
    model: GuardNet,
    rows: Sequence[LabeledInput],
    tok: Tokenizer,
    batch: int = 256,
    temperature: float = 1.0,
) -> dict[Head, FloatArray]:
    model.eval()
    device = next(model.parameters()).device

    out: dict[Head, list[FloatArray]] = {k: [] for k in LABELS}

    with torch.inference_mode():
        for i in range(0, len(rows), batch):
            a, b, _ = tensors(rows[i : i + batch], tok, device)
            outputs = model(a, b)

            for k in out:
                out[k].append(
                    torch.softmax(
                        outputs[k] / (temperature if k == "decision" else 1), dim=-1
                    )
                    .cpu()
                    .numpy()
                )

    return {k: np.concatenate(v) for k, v in out.items()}


def fit_temperature(
    model: GuardNet, rows: Sequence[LabeledInput], tok: Tokenizer, batch: int = 256
) -> float:
    # Held-out validation only; no test-set calibration.
    model.eval()
    device = next(model.parameters()).device
    logits = []
    ys = []

    with torch.inference_mode():
        for i in range(0, len(rows), batch):
            a, b, labels = tensors(rows[i : i + batch], tok, device)
            logits.append(model(a, b)["decision"].cpu())
            ys.append(labels["decision"].cpu())

    z = torch.cat(logits)
    y = torch.cat(ys)
    temperatures = np.linspace(0.45, 3.0, 105)
    vals = [float(nn.functional.cross_entropy(z / t, y)) for t in temperatures]

    return float(temperatures[int(np.argmin(vals))])


def train(
    data: Pathish = "data",
    out: Pathish = "artifacts/guard_best",
    seed: int = 42,
    epochs: int = 8,
    dim: int = 48,
    hidden: int = 96,
    lr: float = 0.002,
    batch: int = 128,
    patience: int = 4,
    device: str = "cpu",
    subword_buckets: int = 0,
    max_len: int = 96,
    max_words: int = 3500,
    attack_weight: float = 0.35,
    nll_tiebreak: bool = False,
    mean_pool: bool = False,
    attention_layers: int = 0,
    context_only: bool = False,
    factorized_policy: bool = False,
    mixed_precision: bool = False,
    stream_features: bool = False,
    accumulation: int = 1,
    validation_batch: int = 256,
    initial_checkpoint: Pathish | None = None,
    gpu_memory_fraction: float | None = None,
) -> dict[str, object]:
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; use the GPU environment or --device cpu")

    if min(epochs, batch, dim, hidden, patience) < 1 or lr <= 0 or attack_weight <= 0:
        raise ValueError(
            "Training dimensions, learning rate and loss weights must be positive"
        )

    if accumulation < 1 or validation_batch < 1:
        raise ValueError("Accumulation and validation batch must be positive")

    if stream_features:
        check_host_memory()

    torch.set_num_threads(1 if stream_features else 2)

    if device == "cuda" and gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tr = load("train", data)
    va = load("val", data)
    tok = Tokenizer(max_words, max_len, subword_buckets, context_only=context_only).fit(
        tr
    )

    heads: tuple[str, ...] = LABELS
    weights = [1.0, 0.20, 0.20, attack_weight]

    if factorized_policy:
        heads = (*LABELS, "directive")
        weights.append(1.0)

    ds: Dataset[tuple[Tensor, ...]]

    if stream_features:
        ds = cache_features(
            tr, tok, Path(out) / "features", directive=factorized_policy
        )
    else:
        ta, tb, tl = tensors(tr, tok)
        targets = [tl[k] for k in LABELS]

        if factorized_policy:
            targets.append(
                torch.tensor(
                    [
                        int(
                            row["injection"]
                            in ("ignore_rules", "secret_request", "override")
                        )
                        for row in tr
                    ],
                    dtype=torch.long,
                )
            )

        ds = TensorDataset(ta, tb, *targets)

    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    model = GuardNet(
        len(tok.stoi),
        dim=dim,
        hidden=hidden,
        subword_buckets=subword_buckets,
        mean_pool=mean_pool,
        attention_layers=attention_layers,
        max_positions=max_len,
        factorized_policy=factorized_policy,
    ).to(device)

    if initial_checkpoint is not None:
        initial = read_checkpoint(initial_checkpoint)

        if initial["config"] != model.config or initial["tokenizer"] != tok.to_dict():
            raise ValueError("Warm-start checkpoint architecture/tokenizer must match")

        model.load_state_dict(initial["state_dict"])
        del initial

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.007)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(epochs, 1), eta_min=lr * 0.06
    )
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    best_key = (-1.0, -float("inf"))
    selected_epoch = 0
    hist = []
    stale = 0
    start = time.time()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for ep in range(epochs):
        model.train()
        losses = []
        optim.zero_grad(set_to_none=True)

        for step, batch_data in enumerate(loader):
            if stream_features and step % 32 == 0:
                check_host_memory()

            a, b, *ys = batch_data
            a, b = a.to(device), b.to(device)
            ys = [y.to(device) for y in ys]

            with torch.autocast(
                device_type=device,
                dtype=torch.bfloat16,
                enabled=mixed_precision and device == "cuda",
            ):
                logits = model(a, b)
                loss = cast(
                    Tensor,
                    sum(
                        w * nn.functional.cross_entropy(logits[k], y)
                        for k, y, w in zip(heads, ys, weights, strict=True)
                    ),
                )

            group_start = (step // accumulation) * accumulation * batch
            group_size = min(accumulation * batch, len(ds) - group_start)
            torch.autograd.backward(loss * a.shape[0] / group_size)

            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

            losses.append(float(loss.detach()))

        sched.step()
        probs = predict_probs(model, va, tok, batch=validation_batch)["decision"]
        y = np.array([CLASSES.index(r["decision"]) for r in va])
        m = score(probs, y)
        rec = {
            "epoch": ep + 1,
            "train_weighted_loss": float(np.mean(losses)),
            "val_accuracy": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
            "val_nll": m["nll"],
            "val_ece": m["ece_10"],
        }
        hist.append(rec)
        print(json.dumps(rec), flush=True)

        key = (m["macro_f1"], -m["nll"])
        improved = key > best_key if nll_tiebreak else m["macro_f1"] > best + 1e-5

        if improved:
            best = m["macro_f1"]
            best_key = key
            selected_epoch = ep + 1
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model.config,
                    "tokenizer": tok.to_dict(),
                    "seed": seed,
                },
                output / "model.tmp",
            )
            (output / "model.tmp").replace(output / "model.pt")
        else:
            stale += 1

            if stale >= patience:
                break

    ck = read_checkpoint(output / "model.pt")
    model.load_state_dict(ck["state_dict"])
    temp = fit_temperature(model, va, tok, batch=validation_batch)
    ck["temperature"] = temp
    torch.save(ck, output / "model.tmp")
    (output / "model.tmp").replace(output / "model.pt")
    res = {
        "config": {
            "seed": seed,
            "epochs_requested": epochs,
            "dim": dim,
            "hidden": hidden,
            "lr": lr,
            "batch": batch,
            "device": device,
            "subword_buckets": subword_buckets,
            "max_len": max_len,
            "max_words": max_words,
            "attack_weight": attack_weight,
            "nll_tiebreak": nll_tiebreak,
            "mean_pool": mean_pool,
            "attention_layers": attention_layers,
            "context_only": context_only,
            "factorized_policy": factorized_policy,
            "mixed_precision": mixed_precision,
            "stream_features": stream_features,
            "accumulation": accumulation,
            "validation_batch": validation_batch,
            "initial_checkpoint": str(initial_checkpoint)
            if initial_checkpoint is not None
            else None,
            "gpu_memory_fraction": gpu_memory_fraction,
        },
        "selected_epoch": selected_epoch,
        "training_device": device,
        "train_examples": len(tr),
        "validation_examples": len(va),
        "best_val_macro_f1": best,
        "temperature": temp,
        "runtime_seconds": round(time.time() - start, 2),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": hist,
    }

    if device == "cuda":
        res["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        res["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20

    (output / "training.json").write_text(json.dumps(res, indent=2))

    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="artifacts/guard_best")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    print(train(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
