"""Tiny from-scratch causal decoder next-token decision baseline.

A genuine causal transformer with teacher labels as next-token targets. Since
only a single class token is needed, generation needs one autoregressive step.
This is NOT a pretrained general instruction LLM; see compare_hf_llm.py.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASSES, load
from .metrics import score
from .model import Tokenizer, tensors
from .types import CausalCheckpoint, CausalConfig, Example, FloatArray, Pathish


class CausalDecisionLM(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim: int = 64,
        layers: int = 2,
        heads: int = 4,
        max_len: int = 96,
    ) -> None:
        super().__init__()

        self.config: CausalConfig = {
            "vocab": vocab,
            "dim": dim,
            "layers": layers,
            "heads": heads,
            "max_len": max_len,
        }
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        block = nn.TransformerEncoderLayer(
            dim, heads, dim * 3, 0.08, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            block, layers, enable_nested_tensor=False
        )
        self.out = nn.Linear(dim, 3)

    def forward(self, tokens: Tensor, fields: Tensor) -> Tensor:
        # The decision is produced at the final prompt position, conditioned on all
        # preceding text. Causal mask prohibits any token from seeing the future.
        batch, length = tokens.shape
        x = (
            self.emb(tokens)
            + self.pos(torch.arange(length, device=tokens.device))[None]
        )
        mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        h = self.transformer(x, mask=mask, src_key_padding_mask=(tokens == 0))
        ix = (tokens != 0).sum(-1).clamp_min(1) - 1

        return cast(Tensor, self.out(h[torch.arange(batch, device=tokens.device), ix]))


def as_prompt_rows(rows: Sequence[Example]) -> list[Example]:
    """Serialize trusted structured state into the autoregressive text context.
    This preserves the same information supplied to the parallel model.
    """
    return [
        {
            **r,
            "context": r["context"]
            + " STATE: "
            + " ".join(
                f"{k}={r[k]}"
                for k in (
                    "action",
                    "sensitivity",
                    "destination",
                    "authorization",
                    "source",
                )
            ),
        }
        for r in rows
    ]


def run(
    data: Pathish = "data",
    out: Pathish = "artifacts/causal",
    epochs: int = 5,
    seed: int = 42,
    dim: int = 64,
    layers: int = 2,
    lr: float = 0.001,
    limit_train: int = 0,
) -> list[dict[str, int | float]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    tr = as_prompt_rows(load("train", data))
    va = as_prompt_rows(load("val", data))

    if limit_train:
        tr = tr[:limit_train]

    tok = Tokenizer(max_len=64).fit(tr)
    a, b, y = tensors(tr, tok)
    ds = TensorDataset(a, b, y["decision"])
    dl = DataLoader(ds, batch_size=96, shuffle=True)
    model = CausalDecisionLM(len(tok.stoi), dim, layers, max_len=tok.max_len)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, int | float]] = []
    best = -1.0

    for epoch in range(epochs):
        model.train()
        losses = []

        for x, f, targets in dl:
            optim.zero_grad()
            logits = model(x, f)
            loss = nn.functional.cross_entropy(logits, targets)
            torch.autograd.backward(loss)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.detach()))

        model.eval()
        allp = []

        with torch.inference_mode():
            for j in range(0, len(va), 96):
                vx, vf, _ = tensors(va[j : j + 96], tok)
                allp.extend(model(vx, vf).softmax(-1).numpy())

        yval = np.array([CLASSES.index(r["decision"]) for r in va])
        m = score(np.array(allp), yval)
        rec = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "val_accuracy": m["accuracy"],
            "val_macro_f1": m["macro_f1"],
        }
        history.append(rec)
        print("causal", json.dumps(rec), flush=True)

        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model.config,
                    "tokenizer": tok.to_dict(),
                },
                path / "model.pt",
            )

    (path / "training.json").write_text(
        json.dumps(
            {
                "history": history,
                "params": sum(p.numel() for p in model.parameters()),
                "note": "From-scratch causal next-class-token model; not a pretrained instruction LLM",
            },
            indent=2,
        )
    )

    return history


def load_causal(path: Pathish) -> tuple[CausalDecisionLM, Tokenizer]:
    ck = cast(
        CausalCheckpoint, torch.load(path, weights_only=False, map_location="cpu")
    )
    m = CausalDecisionLM(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()

    return m, Tokenizer.from_dict(ck["tokenizer"])


def predict(m: CausalDecisionLM, tok: Tokenizer, rows: Sequence[Example]) -> FloatArray:
    prompts = as_prompt_rows(rows)

    allp: list[FloatArray] = []

    with torch.inference_mode():
        for i in range(0, len(prompts), 96):
            x, f, _ = tensors(prompts[i : i + 96], tok)
            allp.extend(m(x, f).softmax(-1).numpy())

    return np.asarray(allp)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="artifacts/causal")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-train", type=int, default=0)
    print(run(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
