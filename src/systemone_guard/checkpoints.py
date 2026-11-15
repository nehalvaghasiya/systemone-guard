"""Load locally trusted guard checkpoints without importing training code."""

from typing import cast

import torch

from .model import GuardNet, Tokenizer
from .types import GuardCheckpoint, Pathish


def read_checkpoint(path: Pathish) -> GuardCheckpoint:
    return cast(
        GuardCheckpoint, torch.load(path, weights_only=False, map_location="cpu")
    )


def load_checkpoint(path: Pathish) -> tuple[GuardNet, Tokenizer, float]:
    checkpoint = read_checkpoint(path)
    model = GuardNet(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tokenizer = Tokenizer.from_dict(checkpoint["tokenizer"])

    return model, tokenizer, checkpoint.get("temperature", 1.0)
