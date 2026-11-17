"""Exercise training, calibration, evaluation, and checkpoint compatibility."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from systemone_guard.checkpoints import load_checkpoint, read_checkpoint
from systemone_guard.data import CLASSES, load
from systemone_guard.evaluate import eval_split
from systemone_guard.inference import DecisionEngine
from systemone_guard.metrics import score
from systemone_guard.model import Tokenizer, tensors
from systemone_guard.robust_train import finalize_checkpoint
from systemone_guard.train import load_checkpoint as legacy_load_checkpoint
from systemone_guard.train import predict_probs, train
from systemone_guard.types import Example, decision_input


@pytest.mark.parametrize("subword_buckets", [0, 32])
def test_training_and_evaluation(
    dataset: Path, tmp_path: Path, subword_buckets: int
) -> None:
    artifacts = tmp_path / "artifacts"
    guard = artifacts / "guard"
    result = train(
        dataset,
        guard,
        seed=7,
        epochs=1,
        dim=8,
        hidden=16,
        batch=16,
        subword_buckets=subword_buckets,
        max_len=48,
        mean_pool=bool(subword_buckets),
        attention_layers=1 if subword_buckets else 0,
        context_only=bool(subword_buckets),
        factorized_policy=bool(subword_buckets),
        stream_features=bool(subword_buckets),
        accumulation=2 if subword_buckets else 1,
        validation_batch=8,
    )
    assert json.loads((guard / "training.json").read_text()) == result
    checkpoint = guard / "model.pt"
    model, tokenizer, temperature = load_checkpoint(checkpoint)
    assert 0.45 <= temperature <= 3.0
    assert result["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert tokenizer.stoi["<pad>"] == 0
    output = tmp_path / "reports"
    scores = eval_split("test", dataset, checkpoint, output)
    assert set(scores) == {"systemone_guard"}
    assert json.loads((output / "test_metrics.json").read_text()) == scores
    records = (output / "test_predictions.jsonl").read_text().splitlines()
    assert len(records) == 24
    assert all(json.loads(line)["id"].startswith("test-") for line in records)


def test_checkpoint_round_trip(
    checkpoint: Path, sample: Example, tmp_path: Path
) -> None:
    assert legacy_load_checkpoint is load_checkpoint
    saved = read_checkpoint(checkpoint)
    tokenizer = Tokenizer.from_dict(saved["tokenizer"])
    assert tokenizer.to_dict() == saved["tokenizer"]
    model, _, temperature = load_checkpoint(checkpoint)
    tokens, fields, _ = tensors([sample], tokenizer)

    with torch.inference_mode():
        expected = model(tokens, fields)["decision"]

    np.testing.assert_array_equal(model(tokens, fields)["decision"].detach(), expected)
    assert temperature == 1.2
    del saved["temperature"]
    old_path = tmp_path / "legacy.pt"
    torch.save(saved, old_path)
    assert load_checkpoint(old_path)[2] == 1.0
    result = DecisionEngine(old_path).decide(**decision_input(sample))
    assert result["temperature"] == 1.0


@pytest.mark.parametrize("mismatch", [False, True])
def test_resource_stop_recovery_checks_validation_before_rewriting_weights(
    checkpoint: Path, dataset: Path, tmp_path: Path, mismatch: bool
) -> None:
    directory = checkpoint.parent
    (directory / "data").symlink_to(dataset, target_is_directory=True)
    (directory / "training.json").write_text(
        json.dumps({"status": "interrupted", "history": [{"epoch": 1}]})
    )
    before = read_checkpoint(checkpoint)
    model, tokenizer, _ = load_checkpoint(checkpoint)
    rows = load("val", dataset)
    metrics = score(
        predict_probs(model, rows, tokenizer)["decision"],
        np.array([CLASSES.index(row["decision"]) for row in rows]),
    )
    log = tmp_path / "training.log"
    log.write_text(
        json.dumps({"train_examples": 96, "validation_examples": len(rows)})
        + "\n"
        + json.dumps(
            {
                "epoch": 1,
                "val_macro_f1": metrics["macro_f1"] + int(mismatch),
                "val_nll": metrics["nll"],
            }
        )
        + "\n"
    )
    report = tmp_path / "recovered.json"

    if mismatch:
        snapshot = checkpoint.read_bytes()

        with pytest.raises(ValueError, match="does not match"):
            finalize_checkpoint(directory, log, report)

        assert checkpoint.read_bytes() == snapshot
        assert not report.exists()
    else:
        result = finalize_checkpoint(directory, log, report)
        after = read_checkpoint(checkpoint)
        assert result["selected_epoch"] == 2
        assert result["runtime_seconds"] is None
        assert json.loads(report.read_text()) == result
        assert before["state_dict"].keys() == after["state_dict"].keys()
        assert all(
            torch.equal(value, after["state_dict"][key])
            for key, value in before["state_dict"].items()
        )
