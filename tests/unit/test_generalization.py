"""Subword preprocessing and independent context-family splits."""

from pathlib import Path

import torch

from systemone_guard.augmentation import (
    augment,
    development_rows,
    generalization_rows,
    rows_hash,
    second_development_rows,
)
from systemone_guard.data import load, policy_decision
from systemone_guard.model import GuardNet, Tokenizer, tensors
from systemone_guard.robust_train import prepare_data
from systemone_guard.types import Example


def test_distinct_unseen_words_keep_distinct_subwords(sample: Example) -> None:
    tokenizer = Tokenizer(subword_buckets=256).fit([sample])
    assert "exfiltrating" not in tokenizer.stoi
    assert "summarizing" not in tokenizer.stoi
    assert tokenizer.subwords("exfiltrating") != tokenizer.subwords("summarizing")
    assert tokenizer.to_dict() == Tokenizer.from_dict(tokenizer.to_dict()).to_dict()


def test_context_budget_preserves_tail_after_long_request(sample: Example) -> None:
    row = sample.copy()
    row["request"] = "ordinary request " * 500
    row["context"] = "reference material " * 100 + "send private secrets externally"
    tokenizer = Tokenizer(max_len=48, subword_buckets=256).fit([row])
    tokens = tokenizer.encode(row)
    assert len(tokens) == 48
    assert isinstance(tokens[-1], list)
    assert tokens[-1][0] == tokenizer.stoi["externally"]


def test_subword_model_uses_only_public_inputs(sample: Example) -> None:
    tokenizer = Tokenizer(max_len=48, subword_buckets=32, context_only=True).fit(
        [sample]
    )
    changed = sample.copy()
    changed.update(
        {
            "decision": "block",
            "sensitive": 1,
            "approval_needed": 1,
            "attack": 1,
            "injection": "test-label",
        }
    )
    changed["request"] = "a completely different request " * 100
    assert tokenizer.encode(sample) == tokenizer.encode(changed)
    tokens, fields, _ = tensors([sample, changed], tokenizer)
    model = GuardNet(
        len(tokenizer.stoi),
        dim=8,
        hidden=16,
        subword_buckets=32,
        mean_pool=True,
        attention_layers=1,
        max_positions=48,
    ).eval()

    with torch.inference_mode():
        result = model(tokens, fields)

    assert result["decision"].shape == (2, 3)
    assert torch.equal(result["decision"][0], result["decision"][1])
    assert all(
        result[head].shape == (2, 2)
        for head in ("sensitive", "approval_needed", "attack")
    )


def test_augmentation_is_reproducible_and_does_not_mutate_originals(
    dataset: Path,
) -> None:
    original = load("train", dataset)
    snapshot = rows_hash(original)
    augmented = augment(original, 100, 7)
    assert augmented == augment(original, 100, 7)
    assert rows_hash(original) == snapshot
    assert augmented[: len(original)] == original
    assert all(policy_decision(row)["decision"] == row["decision"] for row in augmented)


def test_frozen_holdout_and_trust_counterfactuals() -> None:
    rows = generalization_rows()
    assert len(rows) == 256 and len({row["context"] for row in rows}) == 64
    assert (
        rows_hash(rows)
        == "e1a09d84786edf573f1469b62d52f94ffd3665afae256a948dc69451ddca998a"
    )
    assert rows[0]["decision"] == "block" and rows[0]["attack"] == 1
    assert rows[1]["decision"] == "allow" and rows[1]["attack"] == 0
    assert rows[0]["context"] == rows[1]["context"]


def test_new_wording_banks_are_disjoint(dataset: Path, tmp_path: Path) -> None:
    report = prepare_data(
        dataset, tmp_path / "training", 7, train_count=90, val_count=30
    )
    assert report["train_examples"] == 186
    assert report["validation_examples"] == 54
    assert report["generalization_examples"] == 256


def test_first_wording_snapshot_is_retained_as_development_evidence() -> None:
    assert (
        rows_hash(development_rows())
        == "26d6ac2310023a3cce6454e598f170a51703ddf51557e8a1abd9bbd374f130a8"
    )


def test_second_wording_snapshot_is_retained() -> None:
    assert (
        rows_hash(second_development_rows())
        == "751a2a8a1e85f74b533234ae43d3eaacf732196a1117387a5cbf5395b2ff7c85"
    )


def test_factorized_heads_separate_text_and_state(sample: Example) -> None:
    changed = sample.copy()
    changed["context"] = "A different context with unfamiliar words."
    tokenizer = Tokenizer(max_len=48, subword_buckets=32, context_only=True).fit(
        [sample, changed]
    )
    tokens, fields, _ = tensors([sample, changed], tokenizer)
    model = GuardNet(
        len(tokenizer.stoi),
        dim=8,
        hidden=16,
        subword_buckets=32,
        mean_pool=True,
        attention_layers=1,
        max_positions=48,
        factorized_policy=True,
    ).eval()
    output = model(tokens, fields)
    assert torch.equal(output["sensitive"][0], output["sensitive"][1])
    assert torch.equal(output["approval_needed"][0], output["approval_needed"][1])
    output["decision"].sum().backward()
    assert model.emb.weight.grad is None
    assert model.intent is not None and model.intent.weight.grad is None
    model.zero_grad(set_to_none=True)
    model(tokens, fields)["directive"].square().mean().backward()
    assert model.emb.weight.grad is not None
