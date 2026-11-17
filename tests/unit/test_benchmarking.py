"""Offline checks of the fine-tuned adapter and matched benchmark workflow."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from systemone_guard.benchmarking.batching import left_pad
from systemone_guard.benchmarking.comparison import (
    benchmark,
    infer_batch,
    infer_generation,
    infer_typed_llm,
    label_tokens,
    load_records,
)
from systemone_guard.benchmarking.finetune import calibrate, encode_examples
from systemone_guard.benchmarking.types import ScoredGeneration
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
        add_special_tokens: bool,
    ) -> Mapping[str, Tensor]:
        self.prompt = prompt

        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, tokens: Tensor, *, skip_special_tokens: bool) -> str:
        return "block yes yes no"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return [{"allow": 3, "review": 4, "block": 5, " no": 6, " yes": 7}[text]]


class ScoredStub:
    def __init__(self, sequences: Tensor, scores: tuple[Tensor, ...]) -> None:
        self.sequences = sequences
        self.scores = scores


class ModelStub:
    def generate(self, **kwargs: object) -> Tensor | ScoredGeneration:
        inputs = cast(Tensor, kwargs["input_ids"])

        if not kwargs.get("return_dict_in_generate"):
            return torch.cat(
                (inputs, torch.tensor([[5, 7, 7, 6]]).expand(inputs.shape[0], -1)),
                dim=1,
            )

        allowed = cast(
            Callable[[int, Tensor], list[int]], kwargs["prefix_allowed_tokens_fn"]
        )
        sequence = inputs.clone()
        scores = []

        for probabilities in ([0.2, 0.3, 0.5], [0.1, 0.9], [0.25, 0.75], [0.6, 0.4]):
            candidates = allowed(0, sequence[0])
            logits = torch.full((sequence.shape[0], 8), -torch.inf)
            logits[:, candidates] = torch.tensor(probabilities).log()
            scores.append(logits)
            sequence = torch.cat((sequence, logits.argmax(-1).unsqueeze(-1)), dim=1)

        return ScoredStub(sequence, tuple(scores))

    def parameters(self) -> Iterator[Tensor]:
        return iter([torch.zeros(2)])

    def to(self, device: str) -> ModelStub:
        return self

    def eval(self) -> ModelStub:
        return self


def test_generation_and_chat_template_fallback(sample: Example) -> None:
    tokenizer = TokenizerStub()
    label, response, milliseconds, tokens = infer_generation(
        ModelStub(),
        tokenizer,
        sample,
        "cpu",
        8,
    )
    assert label == "block" and response == "block yes yes no" and tokens == 4
    assert milliseconds >= 0
    assert "AUTHORIZATION:" in tokenizer.prompt
    assert "UNTRUSTED_CONTEXT:" in tokenizer.prompt
    assert sample["request"] in tokenizer.prompt


def test_constrained_probabilities_cover_all_four_outputs(sample: Example) -> None:
    tokenizer = TokenizerStub()
    result = infer_typed_llm(
        ModelStub(), tokenizer, sample, "cpu", label_tokens(tokenizer)
    )
    assert result["prediction"] == "block"
    assert result["probabilities"] == pytest.approx(
        {"allow": 0.2, "review": 0.3, "block": 0.5}
    )
    assert result["auxiliary_probabilities"] == pytest.approx(
        {"sensitive": 0.9, "approval_needed": 0.75, "attack": 0.4}
    )
    assert result["new_tokens"] == 4
    assert "sensitive approval_needed attack" in tokenizer.prompt


def test_manual_probes_reuse_fixed_inputs_and_annotations() -> None:
    rows = load_records("manual_probes", 0, 42)
    assert len(rows) == 24
    attacks = [row for row in rows if row["attack"]]
    assert len(attacks) == 10
    assert all(row["decision"] == "block" for row in attacks)
    assert len({row["id"] for row in rows}) == 24


def test_batched_quality_does_not_invent_request_timings(sample: Example) -> None:
    tokenizer = TokenizerStub()
    records = infer_batch(
        ModelStub(),
        tokenizer,
        [sample, sample],
        "cpu",
        label_tokens(tokenizer),
        1.0,
        8,
    )
    assert len(records) == 2
    assert all(record["latency_ms"] is None for record in records)
    assert all(record["generation"]["latency_ms"] is None for record in records)
    assert all(record["prediction"] == "block" for record in records)
    assert records[1]["probabilities"] == pytest.approx(
        {"allow": 0.2, "review": 0.3, "block": 0.5}
    )


def test_left_padding_preserves_real_eos_tokens() -> None:
    inputs = left_pad([torch.tensor([2, 7, 8]), torch.tensor([5, 6])], 2)
    assert inputs["input_ids"].tolist() == [[2, 7, 8], [2, 5, 6]]
    assert inputs["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    assert inputs["position_ids"].tolist() == [[0, 1, 2], [0, 0, 1]]


def test_training_labels_never_change_the_inference_prompt(sample: Example) -> None:
    tokenizer = TokenizerStub()
    original = encode_examples([sample], tokenizer)[0]
    original_prompt = tokenizer.prompt
    changed = sample.copy()
    changed["decision"] = "review"
    changed["sensitive"] = 1
    changed["approval_needed"] = 0
    changed["attack"] = 1
    encoded = encode_examples([changed], tokenizer)[0]
    assert tokenizer.prompt == original_prompt
    assert torch.equal(original.prompt, encoded.prompt)
    assert encoded.answer.tolist() == [4, 7, 6, 7, 1]


def test_validation_temperature_preserves_correct_decisions() -> None:
    probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]])
    temperature, metrics = calibrate(probabilities, [0, 1])
    assert temperature == pytest.approx(0.45)
    assert metrics["accuracy"] == 1
    assert metrics["nll"] < -math.log(0.7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_guard_cpu_cuda_parity(checkpoint: Path, sample: Example) -> None:
    from systemone_guard.inference import DecisionEngine
    from systemone_guard.types import decision_input

    cpu = DecisionEngine(checkpoint).decide(**decision_input(sample))
    cuda = DecisionEngine(checkpoint, device="cuda").decide(**decision_input(sample))
    assert cuda["decision"] == cpu["decision"]
    assert cuda["probabilities"] == pytest.approx(cpu["probabilities"], abs=1e-5)


def test_matched_guard_benchmark_without_downloads(
    dataset: Path,
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "systemone_guard.benchmarking.comparison.load_tuned_model",
        lambda adapter, device: (TokenizerStub(), ModelStub(), 1.0, {}),
    )
    rows = load_records("test", 3, 7, dataset)
    assert rows == load_records("test", 3, 7, dataset)
    assert len(load_records("test", 0, 7, dataset)) == 24
    output = tmp_path / "comparison.json"
    args = argparse.Namespace(
        adapter=tmp_path / "adapter",
        batch_size=2,
        latency_count=2,
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
    assert [record["id"] for record in result["records"]] == [
        row["id"] for row in rows
    ] * 2
    summary = result["summaries"][0]
    assert summary is not None and summary["n"] == 3
    assert summary["invalid_rate"] == 0
