"""Shared record, tensor, metric, and checkpoint contracts.

These annotations describe the existing dictionaries; they do not change their
serialized representation or add inference-time fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
from torch import Tensor

Pathish: TypeAlias = str | Path
FloatArray: TypeAlias = npt.NDArray[np.float32 | np.float64]
StateField: TypeAlias = Literal[
    "action", "sensitivity", "destination", "authorization", "source"
]
Head: TypeAlias = Literal["decision", "sensitive", "approval_needed", "attack"]


class DecisionInput(TypedDict):
    request: str
    context: str
    action: str
    sensitivity: str
    destination: str
    authorization: str
    source: str


def decision_input(row: DecisionInput) -> DecisionInput:
    """Project a labeled record onto the seven public inference arguments."""
    return {
        "request": row["request"],
        "context": row["context"],
        "action": row["action"],
        "sensitivity": row["sensitivity"],
        "destination": row["destination"],
        "authorization": row["authorization"],
        "source": row["source"],
    }


class PolicyLabels(TypedDict):
    decision: str
    sensitive: int
    approval_needed: int
    attack: int


class LabeledInput(DecisionInput, PolicyLabels):
    pass


class UnlabeledExample(DecisionInput):
    id: str
    injection: str


class Example(UnlabeledExample, PolicyLabels):
    pass


class DecisionResult(TypedDict):
    decision: str
    probabilities: dict[str, float]
    confidence: float
    review_recommended: bool
    sensitive_probability: float
    approval_needed_probability: float
    attack_probability: float
    model_type: str
    temperature: float


class TokenizerConfigRequired(TypedDict):
    max_words: int
    max_len: int
    stoi: dict[str, int]


class TokenizerConfig(TokenizerConfigRequired, total=False):
    subword_buckets: int
    subword_slots: int
    context_only: bool


class GuardConfigRequired(TypedDict):
    vocab: int
    dim: int
    hidden: int
    dropout: float


class GuardConfig(GuardConfigRequired, total=False):
    subword_buckets: int
    mean_pool: bool
    attention_layers: int
    max_positions: int
    factorized_policy: bool


class GuardCheckpointRequired(TypedDict):
    state_dict: dict[str, Tensor]
    config: GuardConfig
    tokenizer: TokenizerConfig


class GuardCheckpoint(GuardCheckpointRequired, total=False):
    seed: int
    temperature: float


class MetricValues(TypedDict):
    n: int
    accuracy: float
    macro_f1: float
    nll: float
    brier: float
    ece_10: float
    confusion_matrix: list[list[int]]


class Scores(MetricValues, total=False):
    aux_accuracy: dict[str, float]
