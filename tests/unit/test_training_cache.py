"""Disk-backed features and memory-pressure aborts protect host responsiveness."""

from pathlib import Path

import pytest
import torch

from systemone_guard.model import Tokenizer, tensors
from systemone_guard.training_cache import cache_features, check_host_memory
from systemone_guard.types import Example


def test_cache_preserves_features_and_separates_labels(
    sample: Example, tmp_path: Path
) -> None:
    other = sample.copy()
    other.update({"decision": "block", "attack": 1, "injection": "override"})
    tokenizer = Tokenizer(max_len=24, subword_buckets=32, context_only=True).fit(
        [sample]
    )
    dataset = cache_features(
        [sample, other], tokenizer, tmp_path / "features", directive=True
    )
    tokens, fields, labels = tensors([sample, other], tokenizer)
    first, second = dataset[0], dataset[1]
    assert torch.equal(first[0], tokens[0]) and torch.equal(first[1], fields[0])
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert int(first[2]) == int(labels["decision"][0])
    assert int(second[-1]) == 1
    assert len(dataset) == 2 and len(first) == 7
    stamp = (tmp_path / "features/ready.json").stat().st_mtime_ns
    cached = cache_features(
        [sample, other], tokenizer, tmp_path / "features", directive=True
    )
    assert torch.equal(cached[1][0], second[0])
    assert (tmp_path / "features/ready.json").stat().st_mtime_ns == stamp


def test_low_host_memory_stops_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "read_text", lambda self: "MemAvailable: 1024 kB\n")

    with pytest.raises(RuntimeError, match="Training stopped"):
        check_host_memory()
