"""Typed probabilistic runtime for programmatic agent authorization."""

import argparse
import json

import torch

from .checkpoints import load_checkpoint
from .data import CLASSES
from .model import FIELD_TYPES, encode_single
from .types import DecisionInput, DecisionResult, Pathish


class DecisionEngine:
    def __init__(self, path: Pathish = "artifacts/guard_best/model.pt") -> None:
        self.model, self.tok, self.temperature = load_checkpoint(path)

    def decide(
        self,
        request: str,
        context: str,
        action: str,
        sensitivity: str,
        destination: str,
        authorization: str,
        source: str,
        threshold: float = 0.72,
    ) -> DecisionResult:
        r: DecisionInput = {
            "request": request,
            "context": context,
            "action": action,
            "sensitivity": sensitivity,
            "destination": destination,
            "authorization": authorization,
            "source": source,
        }

        for name, vals in FIELD_TYPES.items():
            if r[name] not in vals:
                raise ValueError(f"{name}: expected one of {vals}")

        # `injection` is not an inference-time field. Model learns to detect untrusted commands from text.
        a, b = encode_single(r, self.tok)

        with torch.inference_mode():
            raw = self.model(a, b)

        p = torch.softmax(raw["decision"] / self.temperature, dim=-1)[0].numpy()
        ix = int(p.argmax())
        decision = CLASSES[ix]

        return {
            "decision": decision,
            "probabilities": {
                k: round(float(v), 6) for k, v in zip(CLASSES, p, strict=True)
            },
            "confidence": round(float(p[ix]), 6),
            "review_recommended": bool(p.max() < threshold or decision == "review"),
            "sensitive_probability": round(
                float(raw["sensitive"].softmax(-1)[0, 1]), 6
            ),
            "approval_needed_probability": round(
                float(raw["approval_needed"].softmax(-1)[0, 1]), 6
            ),
            "attack_probability": round(float(raw["attack"].softmax(-1)[0, 1]), 6),
            "model_type": "single_pass_parallel_typed_decision",
            "temperature": round(self.temperature, 4),
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--request", default="Please read the customer records for a weekly report"
    )
    p.add_argument("--context", default="Tool returned a normal response.")
    p.add_argument("--action", default="read")
    p.add_argument("--sensitivity", default="internal")
    p.add_argument("--destination", default="internal")
    p.add_argument("--authorization", default="approved")
    p.add_argument("--source", default="user")
    p.add_argument("--checkpoint", default="artifacts/guard_best/model.pt")
    args = p.parse_args()
    checkpoint = args.checkpoint
    del args.checkpoint
    print(json.dumps(DecisionEngine(checkpoint).decide(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
