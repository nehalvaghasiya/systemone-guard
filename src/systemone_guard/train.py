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
from torch.utils.data import DataLoader, TensorDataset

from .checkpoints import load_checkpoint as load_checkpoint
from .checkpoints import read_checkpoint
from .data import CLASSES, load
from .metrics import score
from .model import LABELS, GuardNet, Tokenizer, tensors
from .types import FloatArray, Head, LabeledInput, Pathish


def predict_probs(
    model: GuardNet,
    rows: Sequence[LabeledInput],
    tok: Tokenizer,
    batch: int = 256,
    temperature: float = 1.0,
) -> dict[Head, FloatArray]:
    model.eval()

    out: dict[Head, list[FloatArray]] = {k: [] for k in LABELS}

    with torch.inference_mode():
        for i in range(0, len(rows), batch):
            a, b, _ = tensors(rows[i : i + batch], tok)
            outputs = model(a, b)

            for k in out:
                out[k].append(
                    torch.softmax(
                        outputs[k] / (temperature if k == "decision" else 1), dim=-1
                    ).numpy()
                )

    return {k: np.concatenate(v) for k, v in out.items()}


def fit_temperature(
    model: GuardNet, rows: Sequence[LabeledInput], tok: Tokenizer
) -> float:
    # Held-out validation only; no test-set calibration.
    model.eval()
    logits = []
    ys = []

    with torch.inference_mode():
        for i in range(0, len(rows), 256):
            a, b, labels = tensors(rows[i : i + 256], tok)
            logits.append(model(a, b)["decision"])
            ys.append(labels["decision"])

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
) -> dict[str, object]:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    tr = load("train", data)
    va = load("val", data)
    tok = Tokenizer().fit(tr)
    ta, tb, tl = tensors(tr, tok)
    ds = TensorDataset(ta, tb, *[tl[k] for k in LABELS])
    loader = DataLoader(ds, batch_size=batch, shuffle=True)
    model = GuardNet(len(tok.stoi), dim=dim, hidden=hidden)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.007)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=max(epochs, 1), eta_min=lr * 0.06
    )
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    hist = []
    stale = 0
    start = time.time()

    for ep in range(epochs):
        model.train()
        losses = []

        for batch_data in loader:
            a, b, *ys = batch_data
            optim.zero_grad(set_to_none=True)
            logits = model(a, b)
            loss = cast(
                Tensor,
                sum(
                    w * nn.functional.cross_entropy(logits[k], y)
                    for k, y, w in zip(LABELS, ys, [1.0, 0.20, 0.20, 0.35], strict=True)
                ),
            )
            torch.autograd.backward(loss)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.detach()))

        sched.step()
        probs = predict_probs(model, va, tok)["decision"]
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

        if m["macro_f1"] > best + 1e-5:
            best = m["macro_f1"]
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model.config,
                    "tokenizer": tok.to_dict(),
                    "seed": seed,
                },
                output / "model.pt",
            )
        else:
            stale += 1

            if stale >= patience:
                break

    ck = read_checkpoint(output / "model.pt")
    model.load_state_dict(ck["state_dict"])
    temp = fit_temperature(model, va, tok)
    ck["temperature"] = temp
    torch.save(ck, output / "model.pt")
    res = {
        "config": {
            "seed": seed,
            "epochs_requested": epochs,
            "dim": dim,
            "hidden": hidden,
            "lr": lr,
            "batch": batch,
        },
        "best_val_macro_f1": best,
        "temperature": temp,
        "runtime_seconds": round(time.time() - start, 2),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history": hist,
    }
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
    print(train(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
