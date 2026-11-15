"""Measure saved guard and causal decoder CPU batch latency."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from systemone_guard.causal import CausalDecisionLM, load_causal
from systemone_guard.checkpoints import load_checkpoint
from systemone_guard.data import load
from systemone_guard.model import GuardNet, tensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--out", default="experiments/latency.json")
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    m: GuardNet | CausalDecisionLM
    torch.set_num_threads(args.threads)
    rows = load("test", args.data)[: args.count]
    output = {}

    for name, ck in [
        ("systemone_guard", str(Path(args.artifacts) / "guard_best/model.pt")),
        ("causal_decoder_scratch", str(Path(args.artifacts) / "causal/model.pt")),
    ]:
        if name == "systemone_guard":
            m, t, _ = load_checkpoint(ck)
        else:
            m, t = load_causal(ck)

        a, b, _ = tensors(rows, t)

        with torch.inference_mode():
            for _ in range(10):
                m(a, b)

            times = []

            for _ in range(40):
                start = time.perf_counter()
                m(a, b)
                times.append((time.perf_counter() - start) * 1000)

        output[name] = {
            "batch_size": len(rows),
            "batch_p50_ms": float(np.median(times)),
            "batch_p95_ms": float(np.quantile(times, 0.95)),
            "per_item_amortized_p50_ms": float(np.median(times) / len(rows)),
            "parameters": sum(p.numel() for p in m.parameters()),
            "threads": args.threads,
            "hardware": "CPU",
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
