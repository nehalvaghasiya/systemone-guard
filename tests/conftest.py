"""Temporary data and checkpoints keep default tests independent of local assets."""

from pathlib import Path

import pytest
import torch

from systemone_guard.data import generate, load
from systemone_guard.model import GuardNet, Tokenizer
from systemone_guard.types import Example


def pytest_sessionstart(session: pytest.Session) -> None:
    torch.set_num_threads(2)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    generate(directory, {"train": 96, "val": 24, "test": 24, "challenge": 16}, seed=7)

    return directory


@pytest.fixture
def sample(dataset: Path) -> Example:
    return load("test", dataset)[0]


@pytest.fixture
def checkpoint(tmp_path: Path, dataset: Path) -> Path:
    torch.manual_seed(7)
    tokenizer = Tokenizer(max_words=128, max_len=24).fit(load("train", dataset))
    model = GuardNet(len(tokenizer.stoi), dim=8, hidden=16)
    path = tmp_path / "guard/model.pt"
    path.parent.mkdir()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "tokenizer": tokenizer.to_dict(),
            "temperature": 1.2,
        },
        path,
    )

    return path
