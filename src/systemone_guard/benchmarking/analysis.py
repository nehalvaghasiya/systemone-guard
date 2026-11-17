"""Paired accuracy effects and latency savings with explicit denominators."""

import math
import statistics
from collections.abc import Sequence

import numpy as np

from .types import BenchmarkRecord, LatencyEffect, PairComparison

GUARD_METHOD = "SystemOne Guard"
SMOLLM_METHOD = "SmolLM2-135M-Instruct-LoRA"
METHODS = (GUARD_METHOD, SMOLLM_METHOD)


def latency_statistics(times: Sequence[float]) -> dict[str, float]:
    """Use the same linear-interpolated p95 definition throughout the reports."""
    return {
        "median": float(statistics.median(times)),
        "p95": float(np.quantile(times, 0.95, method="linear")),
        "mean": float(statistics.mean(times)),
    }


def _index(
    records: Sequence[BenchmarkRecord], method: str
) -> dict[str, BenchmarkRecord]:
    indexed: dict[str, BenchmarkRecord] = {}

    for record in records:
        if record["method"] != method:
            continue

        identifier = record.get("id")

        if not identifier or identifier in indexed:
            raise ValueError("Comparison records require unique, nonempty IDs")

        indexed[identifier] = record

    return indexed


def _times(records: dict[str, BenchmarkRecord], identifiers: list[str]) -> list[float]:
    result = []

    for identifier in identifiers:
        latency = records[identifier]["latency_ms"]

        if latency is None or not math.isfinite(latency) or latency <= 0:
            raise ValueError("Measured latency must be finite and positive")

        result.append(latency)

    return result


def compare_records(
    records: Sequence[BenchmarkRecord], seed: int = 42
) -> PairComparison:
    """Derive effects from records; do not subtract rounded display percentages."""
    if {record["method"] for record in records} != set(METHODS):
        raise ValueError("Only the guard and the fine-tuned SmolLM2 may be compared")

    guard = _index(records, GUARD_METHOD)
    smollm = _index(records, SMOLLM_METHOD)

    if not guard or set(guard) != set(smollm):
        raise ValueError("Both models must evaluate identical example IDs")

    identifiers = list(guard)

    if any(guard[key]["gold"] != smollm[key]["gold"] for key in identifiers):
        raise ValueError("Gold labels differ between the compared models")

    timed = [key for key in identifiers if guard[key]["latency_ms"] is not None]
    smollm_timed = {key for key in identifiers if smollm[key]["latency_ms"] is not None}

    if not timed or set(timed) != smollm_timed:
        raise ValueError("Both models require the same nonempty timing subset")

    guard_times = _times(guard, timed)
    smollm_times = _times(smollm, timed)
    guard_correct = sum(row["prediction"] == row["gold"] for row in guard.values())
    smollm_correct = sum(row["prediction"] == row["gold"] for row in smollm.values())
    additional = smollm_correct - guard_correct
    delta = np.asarray(
        [
            int(smollm[key]["prediction"] == smollm[key]["gold"])
            - int(guard[key]["prediction"] == guard[key]["gold"])
            for key in identifiers
        ]
    )
    values, counts = np.unique(delta, return_counts=True)
    draws = np.random.default_rng(seed).multinomial(
        len(delta), counts / len(delta), size=10000
    )
    interval = np.quantile(100 * (draws @ values) / len(delta), [0.025, 0.975])

    guard_stats, smollm_stats = (
        latency_statistics(guard_times),
        latency_statistics(smollm_times),
    )

    latency: dict[str, LatencyEffect] = {
        name: {
            "guard_ms": guard_value,
            "smollm_ms": smollm_stats[name],
            "saved_ms": smollm_stats[name] - guard_value,
            "guard_reduction_percent": 100
            * (smollm_stats[name] - guard_value)
            / smollm_stats[name],
            "guard_speedup": smollm_stats[name] / guard_value,
        }
        for name, guard_value in guard_stats.items()
    }

    return {
        "quality_n": len(identifiers),
        "latency_n": len(timed),
        "guard_correct": guard_correct,
        "smollm_correct": smollm_correct,
        "additional_correct": additional,
        "accuracy_gain_pp": 100 * additional / len(identifiers),
        "accuracy_relative_gain_percent": 100 * additional / guard_correct
        if guard_correct
        else None,
        "accuracy_gain_ci95_pp": [float(value) for value in interval],
        "latency": latency,
        "formulas": {
            "accuracy_gain_pp": "100 * (SmolLM2 correct - Guard correct) / quality n",
            "accuracy_relative_gain_percent": "100 * (SmolLM2 correct - Guard correct) / Guard correct; undefined when Guard correct = 0",
            "guard_latency_reduction_percent": "100 * (SmolLM2 latency - Guard latency) / SmolLM2 latency",
            "guard_speedup": "SmolLM2 latency / Guard latency; ratio of the same statistic, not a median of per-example ratios",
        },
    }
