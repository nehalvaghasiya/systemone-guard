"""Explicit validation of original local artifacts; excluded from default tests."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest

from systemone_guard.checkpoints import load_checkpoint
from systemone_guard.data import load
from systemone_guard.inference import DecisionEngine
from systemone_guard.types import DecisionInput, DecisionResult

pytestmark = pytest.mark.artifacts
ROOT = Path(__file__).resolve().parents[2]


class Snapshot(TypedDict):
    checkpoint_sha256: str
    state_keys: list[str]
    examples: list[DecisionInput]
    predictions: list[DecisionResult]


def test_original_checkpoint_predictions() -> None:
    checkpoint = ROOT / "artifacts/guard_best/model.pt"
    assert checkpoint.is_file(), f"Original trained checkpoint required: {checkpoint}"
    snapshot = cast(
        Snapshot,
        json.loads((ROOT / "tests/fixtures/guard_predictions.json").read_text()),
    )
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        == snapshot["checkpoint_sha256"]
    )
    model, _, _ = load_checkpoint(checkpoint)
    assert list(model.state_dict()) == snapshot["state_keys"]
    engine = DecisionEngine(checkpoint)

    for inputs, expected in zip(
        snapshot["examples"], snapshot["predictions"], strict=True
    ):
        actual = engine.decide(**inputs)
        assert actual.keys() == expected.keys()
        assert actual["decision"] == expected["decision"]
        assert actual["review_recommended"] == expected["review_recommended"]
        np.testing.assert_allclose(
            list(actual["probabilities"].values()),
            list(expected["probabilities"].values()),
            rtol=0,
            atol=1e-6,
        )


def test_original_dataset_splits() -> None:
    for split, size in (
        ("train", 10000),
        ("val", 1500),
        ("test", 1500),
        ("challenge", 600),
    ):
        path = ROOT / "data" / f"{split}.jsonl"
        assert path.is_file(), f"Original synthetic dataset required: {path}"
        assert len(load(split, ROOT / "data")) == size
