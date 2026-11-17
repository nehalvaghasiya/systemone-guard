import hashlib
import json
import math
from pathlib import Path

import pytest

from systemone_guard.benchmarking.analysis import (
    GUARD_METHOD,
    SMOLLM_METHOD,
    compare_records,
)
from systemone_guard.benchmarking.comparison import (
    TASK_PROMPT,
    cached_reference,
    parse_typed_label,
    summarize,
    user_prompt,
)
from systemone_guard.benchmarking.types import BenchmarkRecord
from systemone_guard.types import Example


def test_strict_parser() -> None:
    assert parse_typed_label("block yes no yes") == "block"
    assert parse_typed_label("ALLOW NO YES NO\n") == "allow"
    assert parse_typed_label("allow no yes") is None
    assert parse_typed_label("allow no yes no because safe") is None


@pytest.mark.parametrize(
    "change", [None, "model", "input", "prompt", "gold", "id", "duplicate"]
)
def test_frozen_quality_cache_requires_identity_and_drops_timings(
    tmp_path: Path, sample: Example, change: str | None
) -> None:
    details: dict[str, object] = {
        "adapter_sha256": "adapter",
        "revision": "revision",
        "temperature": 0.45,
        "dtype": "float32",
    }
    metadata: dict[str, object] = {
        "records_sha256": hashlib.sha256(
            json.dumps([sample], sort_keys=True).encode()
        ).hexdigest(),
        "task_prompt": TASK_PROMPT,
        "max_new_tokens": 8,
        "quality_batch_size_llm": 4,
        "model_details": {SMOLLM_METHOD: dict(details)},
    }
    record: BenchmarkRecord = {
        "id": sample["id"],
        "method": SMOLLM_METHOD,
        "gold": sample["decision"],
        "prediction": sample["decision"],
        "latency_ms": 12,
        "generation": {
            "prediction": "allow",
            "response": "allow no no no",
            "latency_ms": 15,
            "new_tokens": 4,
        },
    }
    records = [record]

    if change == "model":
        details["temperature"] = 1.0
    elif change == "input":
        metadata["records_sha256"] = "different-input"
    elif change == "prompt":
        metadata["task_prompt"] = "different-prompt"
    elif change == "gold":
        record["gold"] = "wrong-gold"
    elif change == "id":
        record["id"] = "other-id"
    elif change == "duplicate":
        records.append(record)

    (tmp_path / "finetuned_test.json").write_text(
        json.dumps({"metadata": metadata, "records": records})
    )
    result = cached_reference(tmp_path, "test", [sample], details, 8)

    if change is not None:
        assert result is None
    else:
        assert result is not None
        cached, batch_size = result
        assert batch_size == 4
        assert cached[0]["latency_ms"] is None
        assert cached[0]["generation"]["latency_ms"] is None
        assert cached[0]["prediction"] == sample["decision"]


def test_summarize_rejects_invalid() -> None:
    x: list[BenchmarkRecord] = [
        {"method": "llm", "gold": "allow", "prediction": "allow", "latency_ms": 3},
        {"method": "llm", "gold": "block", "prediction": None, "latency_ms": 5},
    ]
    out = summarize(x, "llm")
    assert out is not None
    assert out["accuracy"] == 0.5 and out["invalid_rate"] == 0.5
    assert out["median_latency_ms"] == 4


def test_quality_denominator_is_separate_from_latency_subset() -> None:
    records: list[BenchmarkRecord] = [
        {"method": "llm", "gold": "allow", "prediction": "allow", "latency_ms": 3},
        {"method": "llm", "gold": "block", "prediction": None, "latency_ms": None},
    ]
    result = summarize(records, "llm")
    assert result is not None
    assert result["n"] == 2
    assert result["accuracy"] == 0.5
    assert result["latency_n"] == 1
    assert result["median_latency_ms"] == result["p95_latency_ms"] == 3


def test_prompt_contains_authoritative_fields(sample: Example) -> None:
    text = user_prompt(sample)
    assert "AUTHORIZATION:" in text and "UNTRUSTED_CONTEXT:" in text


def test_probability_and_generation_metrics_remain_separate() -> None:
    records: list[BenchmarkRecord] = [
        {
            "method": "llm",
            "gold": "allow",
            "prediction": "allow",
            "latency_ms": 3,
            "probabilities": {"allow": 0.8, "review": 0.1, "block": 0.1},
            "auxiliary_gold": {"sensitive": 1, "approval_needed": 0, "attack": 0},
            "auxiliary_probabilities": {
                "sensitive": 0.9,
                "approval_needed": 0.5,
                "attack": 0.1,
            },
            "generation": {
                "prediction": None,
                "response": "explanation",
                "latency_ms": 5,
                "new_tokens": 1,
            },
        },
        {
            "method": "llm",
            "gold": "block",
            "prediction": "block",
            "latency_ms": 7,
            "probabilities": {"allow": 0.05, "review": 0.05, "block": 0.9},
            "auxiliary_gold": {"sensitive": 0, "approval_needed": 1, "attack": 1},
            "auxiliary_probabilities": {
                "sensitive": 0.8,
                "approval_needed": 0.9,
                "attack": 0.9,
            },
            "generation": {
                "prediction": "block",
                "response": "BLOCK",
                "latency_ms": 9,
                "new_tokens": 1,
            },
        },
    ]
    summary = summarize(records, "llm")
    assert summary is not None
    assert summary["accuracy"] == 1
    assert summary["nll"] == pytest.approx(-(math.log(0.8) + math.log(0.9)) / 2)
    assert summary["brier"] == pytest.approx(0.0375)
    assert summary["ece_10"] == pytest.approx(0.15)
    assert summary["aux_accuracy"] == {
        "sensitive": 0.5,
        "approval_needed": 1,
        "attack": 1,
    }
    assert summary["generation"]["accuracy"] == 0.5
    assert summary["generation"]["invalid_rate"] == 0.5


def paired_records() -> list[BenchmarkRecord]:
    return [
        {
            "id": "a",
            "method": GUARD_METHOD,
            "gold": "allow",
            "prediction": "allow",
            "latency_ms": 1.0,
        },
        {
            "id": "b",
            "method": GUARD_METHOD,
            "gold": "block",
            "prediction": "review",
            "latency_ms": 3.0,
        },
        {
            "id": "c",
            "method": GUARD_METHOD,
            "gold": "review",
            "prediction": "review",
            "latency_ms": None,
        },
        {
            "id": "a",
            "method": SMOLLM_METHOD,
            "gold": "allow",
            "prediction": "allow",
            "latency_ms": 10.0,
        },
        {
            "id": "b",
            "method": SMOLLM_METHOD,
            "gold": "block",
            "prediction": "block",
            "latency_ms": 20.0,
        },
        {
            "id": "c",
            "method": SMOLLM_METHOD,
            "gold": "review",
            "prediction": "review",
            "latency_ms": None,
        },
    ]


def test_effect_denominators_and_interpolated_percentile() -> None:
    effect = compare_records(paired_records())
    assert effect["quality_n"] == 3 and effect["latency_n"] == 2
    assert effect["additional_correct"] == 1
    assert effect["accuracy_gain_pp"] == pytest.approx(100 / 3)
    assert effect["accuracy_relative_gain_percent"] == 50.0
    assert effect["latency"]["median"]["guard_ms"] == 2.0
    assert effect["latency"]["median"]["smollm_ms"] == 15.0
    assert effect["latency"]["median"]["saved_ms"] == 13.0
    assert effect["latency"]["median"]["guard_reduction_percent"] == pytest.approx(
        100 * 13 / 15
    )
    assert effect["latency"]["median"]["guard_speedup"] == 7.5
    assert effect["latency"]["p95"]["guard_ms"] == pytest.approx(2.9)
    assert effect["latency"]["p95"]["smollm_ms"] == pytest.approx(19.5)


def test_negative_accuracy_and_zero_baseline_are_preserved() -> None:
    records = paired_records()
    records[3]["prediction"] = "block"
    records[4]["prediction"] = "allow"
    effect = compare_records(records)
    assert effect["additional_correct"] == -1
    assert effect["accuracy_gain_pp"] == pytest.approx(-100 / 3)
    assert effect["accuracy_relative_gain_percent"] == -50.0

    for record in records[:3]:
        record["prediction"] = None

    assert compare_records(records)["accuracy_relative_gain_percent"] is None


@pytest.mark.parametrize(
    "change", ["id", "gold", "timing", "duplicate", "model", "negative_latency"]
)
def test_unmatched_or_invalid_comparisons_are_rejected(change: str) -> None:
    records = paired_records()

    if change == "id":
        records[3]["id"] = "other"
    elif change == "gold":
        records[3]["gold"] = "block"
    elif change == "timing":
        records[3]["latency_ms"] = None
    elif change == "duplicate":
        records[3]["id"] = "b"
    elif change == "model":
        records[3]["method"] = "unexpected"
    else:
        records[3]["latency_ms"] = -1.0

    with pytest.raises(ValueError):
        compare_records(records)
