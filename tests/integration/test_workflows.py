"""Exercise training, calibration, evaluation, and checkpoint compatibility."""

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from systemone_guard.baselines import fit
from systemone_guard.causal import load_causal, predict, run
from systemone_guard.checkpoints import load_checkpoint, read_checkpoint
from systemone_guard.data import load
from systemone_guard.evaluate import eval_split
from systemone_guard.inference import DecisionEngine
from systemone_guard.model import Tokenizer, tensors
from systemone_guard.train import load_checkpoint as legacy_load_checkpoint
from systemone_guard.train import train
from systemone_guard.types import Example, decision_input


def test_training_and_evaluation(dataset: Path, tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    guard = artifacts / "guard"
    result = train(dataset, guard, seed=7, epochs=1, dim=8, hidden=16, batch=16)
    assert json.loads((guard / "training.json").read_text()) == result
    checkpoint = guard / "model.pt"
    model, tokenizer, temperature = load_checkpoint(checkpoint)
    assert 0.45 <= temperature <= 3.0
    assert result["parameter_count"] == sum(p.numel() for p in model.parameters())
    assert tokenizer.stoi["<pad>"] == 0
    baseline = fit(load("train", dataset))

    with (artifacts / "word_tfidf.pkl").open("wb") as stream:
        pickle.dump(baseline, stream)

    run(dataset, artifacts / "causal", epochs=1, seed=7, dim=8, layers=1)
    causal, causal_tokenizer = load_causal(artifacts / "causal/model.pt")
    probabilities = predict(causal, causal_tokenizer, load("test", dataset))
    np.testing.assert_allclose(probabilities.sum(1), 1.0, atol=1e-6)
    output = tmp_path / "reports"
    scores = eval_split("test", dataset, checkpoint, output, artifacts=artifacts)
    assert set(scores) == {"systemone_guard", "word_tfidf", "causal_decoder_scratch"}
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
