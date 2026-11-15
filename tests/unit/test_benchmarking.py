"""Offline checks of the pretrained adapter and matched benchmark workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from systemone_guard.benchmarking.pretrained import (
    benchmark,
    infer_llm,
    load_records,
)
from systemone_guard.types import Example


class TokenizerStub:
    pad_token_id: int | None = 0
    pad_token: str | None = "<pad>"
    eos_token: str | None = "</s>"
    eos_token_id: int | None = 1
    prompt = ""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        raise ValueError("This tokenizer has no chat template")

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> Mapping[str, Tensor]:
        self.prompt = prompt

        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, tokens: Tensor, *, skip_special_tokens: bool) -> str:
        return "ALLOW"


class ModelStub:
    def generate(self, **kwargs: object) -> Tensor:
        return torch.tensor([[1, 2, 3]])

    def to(self, device: str) -> ModelStub:
        return self

    def eval(self) -> ModelStub:
        return self


def test_generation_and_chat_template_fallback(sample: Example) -> None:
    tokenizer = TokenizerStub()
    label, response, milliseconds, tokens = infer_llm(
        ModelStub(),
        tokenizer,
        sample,
        "cpu",
        8,
    )
    assert label == "allow" and response == "ALLOW" and tokens == 1
    assert milliseconds >= 0
    assert "AUTHORIZATION:" in tokenizer.prompt
    assert "UNTRUSTED_CONTEXT:" in tokenizer.prompt
    assert sample["request"] in tokenizer.prompt


def test_matched_guard_benchmark_without_downloads(
    dataset: Path,
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    rows = load_records("test", 3, 7, dataset)
    assert rows == load_records("test", 3, 7, dataset)
    assert len(load_records("test", 0, 7, dataset)) == 24
    output = tmp_path / "comparison.json"
    args = argparse.Namespace(
        models=[],
        split="test",
        data=dataset,
        count=3,
        seed=7,
        threads=2,
        device="cpu",
        guard_checkpoint=checkpoint,
        max_new_tokens=8,
        out=output,
    )
    result = benchmark(args)
    assert json.loads(output.read_text()) == result
    assert [record["id"] for record in result["records"]] == [row["id"] for row in rows]
    summary = result["summaries"][0]
    assert summary is not None and summary["n"] == 3
    assert summary["invalid_rate"] == 0
