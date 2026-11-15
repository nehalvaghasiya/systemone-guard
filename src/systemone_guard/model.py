"""Custom parallel typed-decision network. No external pretrained weights."""

from __future__ import annotations

import collections
import re
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

from .data import ACTIONS, AUTH, CLASSES, DESTINATIONS, SENSITIVITY, TRUST
from .types import (
    DecisionInput,
    GuardConfig,
    Head,
    LabeledInput,
    StateField,
    TokenizerConfig,
)

FIELD_TYPES: dict[StateField, list[str]] = {
    "action": ACTIONS,
    "sensitivity": SENSITIVITY,
    "destination": DESTINATIONS,
    "authorization": AUTH,
    "source": TRUST,
}
TEXT_FEATURES: tuple[Literal["request", "context"], ...] = ("request", "context")
LABELS: tuple[Head, ...] = ("decision", "sensitive", "approval_needed", "attack")


class Tokenizer:
    def __init__(self, max_words: int = 3500, max_len: int = 96) -> None:
        self.max_words = max_words
        self.max_len = max_len
        self.stoi = {"<pad>": 0, "<unk>": 1}

    def words(self, s: str) -> list[str]:
        return re.findall(r"[a-z]+|\d+|[^\w\s]", s.lower())

    def fit(self, rows: Sequence[DecisionInput]) -> Tokenizer:
        counter = collections.Counter(
            w for r in rows for k in TEXT_FEATURES for w in self.words(r[k])
        )
        self.stoi.update(
            {
                w: i + 2
                for i, (w, _) in enumerate(counter.most_common(self.max_words - 2))
            }
        )

        return self

    def encode(self, r: DecisionInput) -> list[int]:
        t = "[req] " + r["request"] + " [ctx] " + r["context"]
        ids = [self.stoi.get(w, 1) for w in self.words(t)][: self.max_len]

        return ids + [0] * (self.max_len - len(ids))

    def to_dict(self) -> TokenizerConfig:
        return {"max_words": self.max_words, "max_len": self.max_len, "stoi": self.stoi}

    @classmethod
    def from_dict(cls, d: TokenizerConfig) -> Tokenizer:
        inst = cls(d["max_words"], d["max_len"])
        inst.stoi = d["stoi"]

        return inst


class GuardNet(nn.Module):
    """Shared text encoder + categorical program-state embeddings + 4 simultaneous heads.

    Depthwise convolutions capture token spans,
    masked attention pooling produces fixed-size state, and structured categorical
    embeddings give typed program state. All heads compute in one forward pass.
    """

    def __init__(
        self, vocab: int, dim: int = 48, hidden: int = 96, dropout: float = 0.12
    ) -> None:
        super().__init__()

        self.config: GuardConfig = dict(
            vocab=vocab, dim=dim, hidden=hidden, dropout=dropout
        )
        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.conv3 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.conv5 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.mix = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.attn = nn.Linear(hidden, 1)
        self.fields = nn.ModuleList(
            [nn.Embedding(len(v), 12) for v in FIELD_TYPES.values()]
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden + len(FIELD_TYPES) * 12, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.decision = nn.Linear(hidden, 3)
        self.sensitive = nn.Linear(hidden, 2)
        self.approval = nn.Linear(hidden, 2)
        self.attack = nn.Linear(hidden, 2)

    def forward(self, tokens: Tensor, fields: Tensor) -> dict[Head, Tensor]:
        mask = tokens != 0
        x = self.emb(tokens)
        xt = x.transpose(1, 2)
        feat = torch.cat(
            [
                x,
                torch.nn.functional.gelu(self.conv3(xt).transpose(1, 2)),
                torch.nn.functional.gelu(self.conv5(xt).transpose(1, 2)),
            ],
            -1,
        )
        h = self.mix(feat)
        weights = self.attn(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        pooled = (weights.unsqueeze(-1) * h).sum(1)
        cat = torch.cat([e(fields[:, i]) for i, e in enumerate(self.fields)], -1)
        z = self.fuse(torch.cat([pooled, cat], -1))

        return {
            "decision": self.decision(z),
            "sensitive": self.sensitive(z),
            "approval_needed": self.approval(z),
            "attack": self.attack(z),
        }


def tensors(
    rows: Sequence[LabeledInput], tok: Tokenizer, device: str | torch.device = "cpu"
) -> tuple[Tensor, Tensor, dict[Head, Tensor]]:
    token = torch.tensor([tok.encode(r) for r in rows], dtype=torch.long, device=device)
    fields = torch.tensor(
        [
            [categories.index(r[k]) for k, categories in FIELD_TYPES.items()]
            for r in rows
        ],
        dtype=torch.long,
        device=device,
    )
    labels = {
        k: torch.tensor(
            [CLASSES.index(r[k]) if k == "decision" else int(r[k]) for r in rows],
            dtype=torch.long,
            device=device,
        )
        for k in LABELS
    }

    return token, fields, labels


def encode_single(
    record: DecisionInput, tok: Tokenizer, device: str | torch.device = "cpu"
) -> tuple[Tensor, Tensor]:
    labeled: LabeledInput = {
        **record,
        "decision": "allow",
        "sensitive": 0,
        "approval_needed": 0,
        "attack": 0,
    }
    a, b, _ = tensors(
        [labeled],
        tok,
        device,
    )

    return a, b
