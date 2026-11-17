"""Response-only LoRA training using train/validation data, without test selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import torch
from torch import Tensor

from ..data import load
from ..metrics import score
from ..types import Example, Scores
from .batching import left_pad
from .comparison import (
    AUXILIARY_HEADS,
    BASE_MODEL_ID,
    CLASSES,
    MODEL_REVISION,
    TASK_PROMPT,
    label_tokens,
    load_backbone,
    make_inputs,
)
from .types import CausalLanguageModel, HfTokenizer, ScoredGeneration


class ForwardOutput(Protocol):
    logits: Tensor


class AdaptedModel(CausalLanguageModel, Protocol):
    def __call__(self, **kwargs: object) -> ForwardOutput: ...

    def train(self, mode: bool = True) -> AdaptedModel: ...

    def save_pretrained(self, directory: str, **kwargs: object) -> None: ...


@dataclass
class EncodedExample:
    prompt: Tensor
    answer: Tensor
    decision: int
    auxiliary: dict[str, int]


def encode_examples(
    rows: list[Example], tokenizer: HfTokenizer
) -> list[EncodedExample]:
    candidates = label_tokens(tokenizer)
    result = []

    if tokenizer.eos_token_id is None:
        raise ValueError("Training requires an EOS token")

    for row in rows:
        prompt = make_inputs(tokenizer, row, "cpu", TASK_PROMPT)["input_ids"][0]

        if prompt.numel() >= 768:
            raise ValueError("Training prompt reaches the truncation limit")

        decision = CLASSES.index(row["decision"])
        auxiliary = {
            "sensitive": row["sensitive"],
            "approval_needed": row["approval_needed"],
            "attack": row["attack"],
        }
        answer = [candidates[0][decision]] + [
            candidates[index][auxiliary[head]]
            for index, head in enumerate(AUXILIARY_HEADS, 1)
        ]
        result.append(
            EncodedExample(
                prompt,
                torch.tensor([*answer, tokenizer.eos_token_id]),
                decision,
                auxiliary,
            )
        )

    return result


def validation(
    model: CausalLanguageModel,
    tokenizer: HfTokenizer,
    examples: list[EncodedExample],
    device: str,
    batch_size: int,
) -> tuple[Scores, Tensor]:
    model.eval()
    candidates = label_tokens(tokenizer)
    probabilities = []

    auxiliary: dict[str, list[int]] = {head: [] for head in AUXILIARY_HEADS}
    pad_id = tokenizer.pad_token_id

    if pad_id is None:
        raise ValueError("Validation requires a padding token")

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        inputs = {
            key: value.to(device)
            for key, value in left_pad([row.prompt for row in batch], pad_id).items()
            if key != "position_ids"
        }
        prompt_length = inputs["input_ids"].shape[1]

        def allowed_tokens(
            batch_id: int, tokens: Tensor, prefix_length: int = prompt_length
        ) -> list[int]:
            return candidates[tokens.shape[0] - prefix_length]

        with torch.inference_mode():
            output = cast(
                ScoredGeneration,
                model.generate(
                    **inputs,
                    max_new_tokens=4,
                    min_new_tokens=4,
                    do_sample=False,
                    prefix_allowed_tokens_fn=allowed_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=pad_id,
                ),
            )

        probabilities.append(
            output.scores[0][:, candidates[0]].float().softmax(-1).cpu()
        )

        for index, head in enumerate(AUXILIARY_HEADS, 1):
            predictions = output.scores[index][:, candidates[index]].argmax(-1)
            auxiliary[head].extend(predictions.cpu().tolist())

    probs = torch.cat(probabilities)
    metrics = score(probs.numpy(), [row.decision for row in examples])
    metrics["aux_accuracy"] = {
        head: sum(
            prediction == row.auxiliary[head]
            for prediction, row in zip(auxiliary[head], examples, strict=True)
        )
        / len(examples)
        for head in AUXILIARY_HEADS
    }

    return metrics, probs


def calibrate(probabilities: Tensor, targets: list[int]) -> tuple[float, Scores]:
    """Use the same validation-only temperature range as guard training."""
    logits = probabilities.double().clamp_min(1e-30).log()
    temperatures = np.linspace(0.45, 3.0, 105)
    losses = [
        float(torch.nn.functional.cross_entropy(logits / value, torch.tensor(targets)))
        for value in temperatures
    ]
    temperature = float(temperatures[int(np.argmin(losses))])

    return temperature, score((logits / temperature).softmax(-1).numpy(), targets)


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable; use a CUDA PyTorch environment with GPU access"
        )

    if min(args.epochs, args.batch_size, args.accumulation, args.rank) < 1:
        raise ValueError("epochs, batch size, accumulation and rank must be positive")

    from peft import LoraConfig, PeftModel, get_peft_model

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    best = output / "best"
    training_rows = load("train", args.data)
    validation_rows = load("val", args.data)

    if args.train_count:
        training_rows = training_rows[: args.train_count]

    if args.val_count:
        validation_rows = validation_rows[: args.val_count]

    started = time.perf_counter()
    precision = "bfloat16" if args.device == "cuda" else "float32"
    tokenizer, base = load_backbone(BASE_MODEL_ID, args.device, precision)
    training = encode_examples(training_rows, tokenizer)
    held_out = encode_examples(validation_rows, tokenizer)
    model = cast(
        AdaptedModel,
        get_peft_model(
            base,
            LoraConfig(
                r=args.rank,
                lora_alpha=2 * args.rank,
                lora_dropout=0.0,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
                revision=MODEL_REVISION,
            ),
        ),
    )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    effective_batch = args.batch_size * args.accumulation
    steps_per_epoch = math.ceil(len(training) / effective_batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.1))

    def learning_rate(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup

        return float(max(0.0, (total_steps - step) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)

    history: list[dict[str, object]] = []
    report: dict[str, object] = {
        "status": "running",
        "base_model": BASE_MODEL_ID,
        "revision": MODEL_REVISION,
        "seed": args.seed,
        "train_examples": len(training),
        "validation_examples": len(held_out),
        "train_sha256": hashlib.sha256(
            json.dumps(training_rows, sort_keys=True).encode()
        ).hexdigest(),
        "validation_sha256": hashlib.sha256(
            json.dumps(validation_rows, sort_keys=True).encode()
        ).hexdigest(),
        "base_parameters": sum(parameter.numel() for parameter in base.parameters())
        - sum(parameter.numel() for parameter in trainable),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "rank": args.rank,
        "lora_alpha": 2 * args.rank,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.accumulation,
        "learning_rate": args.lr,
        "training_precision": precision,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "peft_version": importlib.metadata.version("peft"),
        "cuda_version": torch.version.cuda,
        "selection": "validation macro-F1, with lower validation NLL breaking ties",
        "loss": "cross entropy on four response tokens and EOS; prompt/padding excluded",
        "history": history,
    }
    best_key = (-1.0, -float("inf"))
    best_probabilities = torch.empty(0)
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        order = torch.randperm(len(training)).tolist()
        loss_sum = 0.0

        for step, start in enumerate(range(0, len(order), effective_batch), 1):
            group = [
                training[index] for index in order[start : start + effective_batch]
            ]
            optimizer.zero_grad(set_to_none=True)

            for offset in range(0, len(group), args.batch_size):
                batch = group[offset : offset + args.batch_size]
                sequences = [torch.cat((row.prompt, row.answer[:-1])) for row in batch]
                inputs = {
                    key: value.to(args.device)
                    for key, value in left_pad(sequences, pad_id).items()
                }
                targets = torch.stack([row.answer for row in batch]).to(args.device)

                with torch.autocast(
                    device_type=args.device,
                    dtype=torch.bfloat16,
                    enabled=args.device == "cuda",
                ):
                    logits = model(**inputs, logits_to_keep=5, use_cache=False).logits
                    loss = torch.nn.functional.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="sum",
                    )

                loss_sum += float(loss.detach())
                torch.autograd.backward(loss / (len(group) * 5))

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

            if step % 25 == 0 or step == steps_per_epoch:
                print(
                    f"epoch {epoch}/{args.epochs}, step {step}/{steps_per_epoch}, response loss {loss_sum / min(start + effective_batch, len(training)) / 5:.4f}",
                    flush=True,
                )

        metrics, probabilities = validation(
            model, tokenizer, held_out, args.device, args.batch_size * 2
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / len(training) / 5,
                "seconds": time.perf_counter() - epoch_start,
                "validation": metrics,
            }
        )
        key = (metrics["macro_f1"], -metrics["nll"])

        if key > best_key:
            best_key = key
            best_probabilities = probabilities
            model.save_pretrained(
                str(best), safe_serialization=True, save_embedding_layers=False
            )
            # The actual training report supplies provenance instead of PEFT's blank card.
            (best / "README.md").unlink(missing_ok=True)
            report["selected_epoch"] = epoch
            report["best_validation"] = metrics

        (output / "training.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(history[-1]), flush=True)

    fp32_tokenizer, fp32_base = load_backbone(BASE_MODEL_ID, args.device)
    selected = PeftModel.from_pretrained(fp32_base, best)
    fp32_model = cast(CausalLanguageModel, selected.merge_and_unload()).eval()
    selected_metrics, best_probabilities = validation(
        fp32_model, fp32_tokenizer, held_out, args.device, args.batch_size * 2
    )
    report["selected_fp32_validation"] = selected_metrics
    temperature, calibrated = calibrate(
        best_probabilities, [row.decision for row in held_out]
    )
    (best / "calibration.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "split": "val",
                "n": len(held_out),
                "metrics": calibrated,
            },
            indent=2,
        )
    )
    report.update(
        status="complete",
        temperature=temperature,
        calibrated_validation=calibrated,
        seconds=time.perf_counter() - started,
        completed_utc=datetime.now(timezone.utc).isoformat(),
    )

    if args.device == "cuda":
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20

    (output / "training.json").write_text(json.dumps(report, indent=2))
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="artifacts/smollm2_lora")
    parser.add_argument("--report", default="experiments/smollm2_finetuning.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-count",
        type=int,
        default=0,
        help="Limit training for smoke checks only",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=0,
        help="Limit validation for smoke checks only",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
