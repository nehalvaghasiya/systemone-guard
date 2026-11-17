"""Shared left padding for response-only training and batched generation."""

import torch
from torch import Tensor


def left_pad(sequences: list[Tensor], pad_id: int) -> dict[str, Tensor]:
    """Mask by length because EOS can be a valid token inside chat prompts."""
    length = max(sequence.numel() for sequence in sequences)
    tokens = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(tokens)

    for index, sequence in enumerate(sequences):
        tokens[index, -sequence.numel() :] = sequence
        mask[index, -sequence.numel() :] = 1

    return {
        "input_ids": tokens,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
    }
