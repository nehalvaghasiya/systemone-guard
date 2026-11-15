"""Contracts at the optional Hugging Face integration boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, TypedDict

from torch import Tensor


class PretrainedTokenizer(Protocol):
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
        self, prompt: str, *, return_tensors: str, truncation: bool, max_length: int
    ) -> Mapping[str, Tensor]: ...

    def decode(self, tokens: Tensor, *, skip_special_tokens: bool) -> str: ...


class PretrainedModel(Protocol):
    def generate(self, **kwargs: object) -> Tensor: ...

    def to(self, device: str) -> PretrainedModel: ...

    def eval(self) -> PretrainedModel: ...


class RecordRequired(TypedDict):
    method: str
    gold: str
    prediction: str | None
    latency_ms: float


class BenchmarkRecord(RecordRequired, total=False):
    id: str | None
    probabilities: dict[str, float]
    response: str
    new_tokens: int


class Summary(TypedDict):
    method: str
    n: int
    accuracy: float
    macro_f1: float
    invalid_rate: float
    median_latency_ms: float
    p95_latency_ms: float
    mean_latency_ms: float
    confusion_matrix: list[list[int]]


class BenchmarkResult(TypedDict):
    metadata: dict[str, object]
    summaries: list[Summary | None]
    records: list[BenchmarkRecord]
