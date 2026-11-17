"""Custom parallel typed-decision network. No external pretrained weights."""

from __future__ import annotations

import collections
import re
import zlib
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
    def __init__(
        self,
        max_words: int = 3500,
        max_len: int = 96,
        subword_buckets: int = 0,
        subword_slots: int = 8,
        context_only: bool = False,
    ) -> None:
        if max_words < 4 or max_len < 4 or subword_buckets < 0 or subword_slots < 2:
            raise ValueError("Invalid tokenizer dimensions")

        self.max_words = max_words
        self.max_len = max_len
        self.subword_buckets = subword_buckets
        self.subword_slots = subword_slots
        self.context_only = context_only
        self.stoi = {"<pad>": 0, "<unk>": 1}

    def words(self, s: str) -> list[str]:
        return re.findall(r"[a-z]+|\d+|[^\w\s]", s.lower())

    def fit(self, rows: Sequence[DecisionInput]) -> Tokenizer:
        counter = collections.Counter(
            w
            for r in rows
            for k in (("context",) if self.context_only else TEXT_FEATURES)
            for w in self.words(r[k])
        )
        self.stoi = {"<pad>": 0, "<unk>": 1}

        if self.subword_buckets:
            self.stoi.update({"<request>": 2, "<context>": 3})

        offset = len(self.stoi)
        self.stoi.update(
            {
                w: i + offset
                for i, (w, _) in enumerate(counter.most_common(self.max_words - offset))
            }
        )

        return self

    def subwords(self, word: str) -> list[int]:
        """Stable character n-grams; never collapse an unfamiliar word to UNK alone."""
        text = f"<{word}>"
        grams = [
            text[start : start + size]
            for size in (3, 4, 5)
            for start in range(max(0, len(text) - size + 1))
        ] or [text]

        if len(grams) > self.subword_slots:
            grams = [
                grams[round(index * (len(grams) - 1) / (self.subword_slots - 1))]
                for index in range(self.subword_slots)
            ]

        ids = [1 + zlib.crc32(gram.encode()) % self.subword_buckets for gram in grams]

        return ids + [0] * (self.subword_slots - len(ids))

    def encode(self, r: DecisionInput) -> list[int] | list[list[int]]:
        if self.subword_buckets:
            request = (
                []
                if self.context_only
                else self.words(r["request"])[: min(32, self.max_len // 4)]
            )
            context = self.words(r["context"])
            budget = self.max_len - len(request) - 2

            if len(context) > budget:
                context = context[: budget // 2] + context[-(budget - budget // 2) :]

            words = ["<request>", *request, "<context>", *context]
            encoded = [[self.stoi.get(word, 1), *self.subwords(word)] for word in words]

            return encoded + [
                [0] * (self.subword_slots + 1)
                for _ in range(self.max_len - len(encoded))
            ]

        t = "[req] " + r["request"] + " [ctx] " + r["context"]
        ids = [self.stoi.get(w, 1) for w in self.words(t)][: self.max_len]

        return ids + [0] * (self.max_len - len(ids))

    def to_dict(self) -> TokenizerConfig:
        config: TokenizerConfig = {
            "max_words": self.max_words,
            "max_len": self.max_len,
            "stoi": self.stoi,
        }

        if self.subword_buckets:
            config["subword_buckets"] = self.subword_buckets
            config["subword_slots"] = self.subword_slots

            if self.context_only:
                config["context_only"] = True

        return config

    @classmethod
    def from_dict(cls, d: TokenizerConfig) -> Tokenizer:
        inst = cls(
            d["max_words"],
            d["max_len"],
            d.get("subword_buckets", 0),
            d.get("subword_slots", 8),
            d.get("context_only", False),
        )
        inst.stoi = d["stoi"]

        return inst


class GuardNet(nn.Module):
    """Shared text encoder + categorical program-state embeddings + 4 simultaneous heads.

    Depthwise convolutions capture token spans,
    masked attention pooling produces fixed-size state, and structured categorical
    embeddings give typed program state. All heads compute in one forward pass.
    """

    def __init__(
        self,
        vocab: int,
        dim: int = 48,
        hidden: int = 96,
        dropout: float = 0.12,
        subword_buckets: int = 0,
        mean_pool: bool = False,
        attention_layers: int = 0,
        max_positions: int = 160,
        factorized_policy: bool = False,
    ) -> None:
        super().__init__()

        self.config: GuardConfig = dict(
            vocab=vocab, dim=dim, hidden=hidden, dropout=dropout
        )

        if subword_buckets:
            self.config["subword_buckets"] = subword_buckets

        if mean_pool:
            self.config["mean_pool"] = True

        self.mean_pool = mean_pool
        self.factorized_policy = factorized_policy

        if factorized_policy:
            self.config["factorized_policy"] = True

        if attention_layers:
            self.config.update(
                {"attention_layers": attention_layers, "max_positions": max_positions}
            )

        self.emb = nn.Embedding(vocab, dim, padding_idx=0)
        self.subemb = (
            nn.Embedding(subword_buckets + 1, dim, padding_idx=0)
            if subword_buckets
            else None
        )
        self.conv3 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.conv5 = nn.Conv1d(dim, dim, 5, padding=2, groups=dim)
        self.mix = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.positions = (
            nn.Embedding(max_positions, hidden) if attention_layers else None
        )
        self.global_encoder = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    hidden,
                    4,
                    hidden * 2,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                ),
                attention_layers,
                enable_nested_tensor=False,
            )
            if attention_layers
            else None
        )

        if factorized_policy and self.positions is not None:
            nn.init.normal_(self.positions.weight, std=0.02)

        if factorized_policy and self.global_encoder is not None:
            for layer in self.global_encoder.layers:
                for name, parameter in layer.named_parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
                    elif name.endswith("weight"):
                        nn.init.ones_(parameter)
                    else:
                        nn.init.zeros_(parameter)

        self.attn = nn.Linear(hidden, 1)
        self.fields = nn.ModuleList(
            [nn.Embedding(len(v), 12) for v in FIELD_TYPES.values()]
        )
        text_width = hidden * (2 if mean_pool else 1)
        self.intent = nn.Linear(text_width, 2) if factorized_policy else None
        self.state_fuse = (
            nn.Sequential(
                nn.Linear(len(FIELD_TYPES) * 12, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            if factorized_policy
            else None
        )
        self.fuse = nn.Sequential(
            nn.Linear(
                (2 if factorized_policy else text_width) + len(FIELD_TYPES) * 12, hidden
            ),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
        )
        self.decision = nn.Linear(hidden, 3)
        self.sensitive = nn.Linear(hidden, 2)
        self.approval = nn.Linear(hidden, 2)
        self.attack = nn.Linear(14 if factorized_policy else hidden, 2)

    def forward(self, tokens: Tensor, fields: Tensor) -> dict[str, Tensor]:
        words = tokens[..., 0] if self.subemb is not None else tokens
        mask = words != 0

        if self.subemb is not None and self.training:
            words = words.masked_fill(
                (torch.rand_like(words, dtype=torch.float) < 0.1) & mask, 1
            )

        x = self.emb(words)

        if self.subemb is not None:
            grams = tokens[..., 1:]
            counts = (grams != 0).sum(-1, keepdim=True).clamp_min(1)
            x = x + self.subemb(grams).sum(-2) / counts

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

        if self.global_encoder is not None and self.positions is not None:
            position = torch.arange(h.shape[1], device=h.device)
            h = self.global_encoder(
                h + self.positions(position), src_key_padding_mask=~mask
            )

        weights = self.attn(h).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        pooled = (weights.unsqueeze(-1) * h).sum(1)

        if self.mean_pool:
            average = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(
                1, keepdim=True
            ).clamp_min(1)
            pooled = torch.cat((pooled, average), -1)

        cat = torch.cat([e(fields[:, i]) for i, e in enumerate(self.fields)], -1)

        if self.intent is not None and self.state_fuse is not None:
            intent = self.intent(pooled)
            # Text learns the independent context label; state tasks cannot distort it.
            evidence = intent.softmax(-1).detach()
            state = self.state_fuse(cat)
            z = self.fuse(torch.cat([evidence, cat], -1))
            source = self.fields[-1](fields[:, -1])

            return {
                "decision": self.decision(z),
                "sensitive": self.sensitive(state),
                "approval_needed": self.approval(state),
                "attack": self.attack(torch.cat([evidence, source], -1)),
                "directive": intent,
            }

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
    shape = (
        (len(rows), tok.max_len, tok.subword_slots + 1)
        if tok.subword_buckets
        else (len(rows), tok.max_len)
    )
    token = torch.empty(shape, dtype=torch.long, device=device)

    for start in range(0, len(rows), 256):
        token[start : start + 256] = torch.tensor(
            [tok.encode(row) for row in rows[start : start + 256]],
            dtype=torch.long,
            device=device,
        )

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
