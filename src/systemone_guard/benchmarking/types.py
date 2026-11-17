"""Contracts at the optional Hugging Face integration boundary."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol, TypedDict

from torch import Tensor


class HfTokenizer(Protocol):
    pad_token_id: int | None
    pad_token: str | None
    eos_token: str | None
    eos_token_id: int | None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
        add_special_tokens: bool,
    ) -> Mapping[str, Tensor]: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, tokens: Tensor, *, skip_special_tokens: bool) -> str: ...


class ScoredGeneration(Protocol):
    sequences: Tensor
    scores: tuple[Tensor, ...]


class CausalLanguageModel(Protocol):
    def generate(self, **kwargs: object) -> Tensor | ScoredGeneration: ...

    def parameters(self) -> Iterator[Tensor]: ...

    def to(self, device: str) -> CausalLanguageModel: ...

    def eval(self) -> CausalLanguageModel: ...


class RecordRequired(TypedDict):
    method: str
    gold: str
    prediction: str | None
    latency_ms: float | None


class BenchmarkRecord(RecordRequired, total=False):
    id: str | None
    probabilities: dict[str, float]
    response: str
    new_tokens: int
    auxiliary_gold: dict[str, int]
    auxiliary_probabilities: dict[str, float]
    generation: GenerationRecord
    group: str


class GenerationRecord(TypedDict):
    prediction: str | None
    response: str
    latency_ms: float | None
    new_tokens: int


class SummaryRequired(TypedDict):
    method: str
    n: int
    accuracy: float
    macro_f1: float
    invalid_rate: float
    median_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    confusion_matrix: list[list[int]]


class Summary(SummaryRequired, total=False):
    latency_n: int
    nll: float
    brier: float
    ece_10: float
    aux_accuracy: dict[str, float]
    per_class: dict[str, dict[str, float]]
    generation: SummaryRequired
    by_group: dict[str, dict[str, float]]


class LatencyEffect(TypedDict):
    guard_ms: float
    smollm_ms: float
    saved_ms: float
    guard_reduction_percent: float
    guard_speedup: float


class PairComparison(TypedDict):
    quality_n: int
    latency_n: int
    guard_correct: int
    smollm_correct: int
    additional_correct: int
    accuracy_gain_pp: float
    accuracy_relative_gain_percent: float | None
    accuracy_gain_ci95_pp: list[float]
    latency: dict[str, LatencyEffect]
    formulas: dict[str, str]


class BenchmarkResult(TypedDict):
    metadata: dict[str, object]
    summaries: list[Summary | None]
    records: list[BenchmarkRecord]
    comparison: PairComparison
