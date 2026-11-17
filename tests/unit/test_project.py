from pathlib import Path

import numpy as np
import pytest
import torch

from systemone_guard.data import CLASSES, load, policy_decision
from systemone_guard.inference import DecisionEngine
from systemone_guard.metrics import score
from systemone_guard.model import GuardNet, Tokenizer, tensors
from systemone_guard.train import load_checkpoint
from systemone_guard.types import decision_input


def test_schema(dataset: Path) -> None:
    rows = load("train", dataset)[:20]
    assert set(r["decision"] for r in rows).issubset(set(CLASSES))
    assert all(policy_decision(r)["decision"] == r["decision"] for r in rows)


def test_disjoint_ids(dataset: Path) -> None:
    assert not (
        {r["id"] for r in load("train", dataset)}
        & {r["id"] for r in load("test", dataset)}
    )


def test_no_label_leak(dataset: Path) -> None:
    inputs = decision_input(load("train", dataset)[0])
    assert "injection" not in inputs and "decision" not in inputs


def test_shapes(dataset: Path) -> None:
    rows = load("train", dataset)[:3]
    t = Tokenizer().fit(rows)
    x, f, _y = tensors(rows, t)
    m = GuardNet(len(t.stoi))
    o = m(x, f)
    assert o["decision"].shape == (3, 3) and all(
        o[k].shape == (3, 2) for k in ("sensitive", "approval_needed", "attack")
    )


def test_probabilities(dataset: Path, checkpoint: Path) -> None:
    m, t, temp = load_checkpoint(checkpoint)
    x, f, _ = tensors(load("test", dataset)[:5], t)

    with torch.no_grad():
        p = (m(x, f)["decision"] / temp).softmax(-1)

    assert torch.allclose(p.sum(1), torch.ones(5), atol=1e-5)


def test_typed_outputs(dataset: Path, checkpoint: Path) -> None:
    r = load("test", dataset)[0]
    e = DecisionEngine(checkpoint)
    d = e.decide(**decision_input(r))
    assert d["decision"] in CLASSES and set(d["probabilities"]) == set(CLASSES)
    assert abs(sum(d["probabilities"].values()) - 1) < 1e-4


def test_reject_invalid_state(checkpoint: Path) -> None:
    with pytest.raises(ValueError):
        DecisionEngine(checkpoint).decide(
            "hi", "hi", "invalid", "public", "internal", "approved", "user"
        )


def test_metrics_perfect() -> None:
    y = np.array([0, 1, 2])
    p = np.eye(3) * 0.999 + (0.001 / 3)
    m = score(p, y)
    assert m["accuracy"] == 1 and m["brier"] < 0.001


def test_attack_policy(dataset: Path) -> None:
    r = load("train", dataset)[0].copy()
    r.update(
        {
            "action": "read",
            "sensitivity": "public",
            "destination": "internal",
            "authorization": "approved",
            "source": "tool_output",
            "injection": "override",
        }
    )
    assert policy_decision(r)["decision"] == "block"


def test_quoted_warning_not_attack(dataset: Path) -> None:
    r = load("train", dataset)[0].copy()
    r.update(
        {
            "action": "read",
            "sensitivity": "public",
            "destination": "internal",
            "authorization": "approved",
            "source": "tool_output",
            "injection": "quoted_warning",
        }
    )
    assert policy_decision(r)["decision"] == "allow"


def test_denied_overrides(dataset: Path) -> None:
    r = load("train", dataset)[0].copy()
    r["authorization"] = "denied"
    assert policy_decision(r)["decision"] == "block"
