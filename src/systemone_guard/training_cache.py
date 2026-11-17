"""Disk-backed token features so larger training sets do not fill host RAM."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import CLASSES
from .model import FIELD_TYPES, LABELS, Tokenizer
from .types import Example


def check_host_memory(minimum_mib: int = 1536) -> None:
    """Stop our worker before consuming the memory needed by the editor/OS."""
    info = Path("/proc/meminfo")

    if not info.exists():
        return

    available = next(
        int(line.split()[1])
        for line in info.read_text().splitlines()
        if line.startswith("MemAvailable:")
    )

    if available < minimum_mib * 1024:
        raise RuntimeError(
            f"Training stopped: less than {minimum_mib} MiB host memory available"
        )


class FeatureDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, directory: Path) -> None:
        self.tokens = cast(
            npt.NDArray[np.uint16], np.load(directory / "tokens.npy", mmap_mode="r")
        )
        self.fields = cast(
            npt.NDArray[np.uint8], np.load(directory / "fields.npy", mmap_mode="r")
        )
        self.targets = cast(
            npt.NDArray[np.uint8], np.load(directory / "targets.npy", mmap_mode="r")
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        # Convert only one row; the loader stacks a bounded microbatch.
        tokens = torch.from_numpy(np.array(self.tokens[index], dtype=np.int64))
        fields = torch.from_numpy(np.array(self.fields[index], dtype=np.int64))
        targets = torch.from_numpy(np.array(self.targets[index], dtype=np.int64))

        return (tokens, fields, *targets.unbind())


def cache_features(
    rows: Sequence[Example], tokenizer: Tokenizer, directory: Path, *, directive: bool
) -> FeatureDataset:
    if max(len(tokenizer.stoi), tokenizer.subword_buckets + 1) > 65535:
        raise ValueError("The compact feature cache requires token IDs below 65536")

    digest = hashlib.sha256(json.dumps(tokenizer.to_dict(), sort_keys=True).encode())

    for row in rows:
        digest.update(json.dumps(row, sort_keys=True).encode())
        digest.update(b"\n")

    key = f"{digest.hexdigest()}-directive-{directive}"
    ready = directory / "ready.json"

    if ready.exists() and json.loads(ready.read_text())["key"] == key:
        return FeatureDataset(directory)

    check_host_memory()
    directory.mkdir(parents=True, exist_ok=True)
    ready.unlink(missing_ok=True)
    shape = (
        (len(rows), tokenizer.max_len, tokenizer.subword_slots + 1)
        if tokenizer.subword_buckets
        else (len(rows), tokenizer.max_len)
    )
    tokens = np.lib.format.open_memmap(
        directory / "tokens.npy", mode="w+", dtype=np.uint16, shape=shape
    )
    fields = np.lib.format.open_memmap(
        directory / "fields.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), len(FIELD_TYPES)),
    )
    targets = np.lib.format.open_memmap(
        directory / "targets.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), len(LABELS) + int(directive)),
    )

    for start in range(0, len(rows), 256):
        check_host_memory()
        batch = rows[start : start + 256]
        tokens[start : start + len(batch)] = np.asarray(
            [tokenizer.encode(row) for row in batch], dtype=np.uint16
        )
        fields[start : start + len(batch)] = np.asarray(
            [
                [values.index(row[name]) for name, values in FIELD_TYPES.items()]
                for row in batch
            ],
            dtype=np.uint8,
        )
        labels = [
            [
                CLASSES.index(row["decision"]),
                row["sensitive"],
                row["approval_needed"],
                row["attack"],
            ]
            for row in batch
        ]

        if directive:
            for values, row in zip(labels, batch, strict=True):
                values.append(
                    int(
                        row["injection"]
                        in ("ignore_rules", "secret_request", "override")
                    )
                )

        targets[start : start + len(batch)] = np.asarray(labels, dtype=np.uint8)

    tokens.flush()
    fields.flush()
    targets.flush()
    del tokens, fields, targets
    ready.write_text(
        json.dumps({"key": key, "rows": len(rows), "dtype": "uint16"}) + "\n"
    )

    return FeatureDataset(directory)
