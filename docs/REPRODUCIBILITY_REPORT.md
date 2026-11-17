# Reproducibility record

The current comparison uses **SystemOne Guard with 756,248 parameters**, all trained from random initialization, and the existing task-fine-tuned **SmolLM2-135M-Instruct LoRA** adapter. Completed epochs and per-example measurements supply the figures. The manual-probe target is met; the separate wording-set target remains unmet.

## Selected guard

The default checkpoint is `artifacts/guard_scaled/model.pt`, SHA-256 `24563b99b1bbacf4cfb555d88f16f2074f934cd5c1da3826781af3279010e5ec`. The [training record](../experiments/guard_training.json) contains configuration, data hashes, completed epoch histories, selected epoch, validation metrics, and temperature. Interrupted-run records remain in [the experiment audit](../experiments/guard_improvement.json).

The encoder has 64-dimensional word/character-subword embeddings, depthwise convolutions, a 128-dimensional projection, three four-head attention layers, and attention/mean pooling. Character n-grams use 4,096 stable CRC32 buckets and eight slots per word. The encoder retains up to 160 context tokens, including both ends of longer text. Request wording is excluded because typed state is authoritative in the simulated policy; the public request argument is retained.

A private context-intent head learns source-independent instruction risk. Its detached probabilities and categorical state embeddings feed learned policy heads. Sensitive and approval heads read typed state; the attack head also receives source trust. Four public outputs retain the original schema. Training annotations supervise the private objective but are never inference inputs. Guard uses no pretrained embeddings or language-model weights.

Training combines the original 10,000 examples with 80,000 augmented examples. Validation combines the original 1,500 with 2,000 separately worded examples. The `context-families-v6` added training bank has 37,401 unique contexts; the new validation bank has 51. No complete context string overlaps between the new train bank, new validation bank, final wording bank, or existing manual probes. The original synthetic splits already shared templates by design. More examples do not imply equally many independent contexts.

The larger model completed five initial epochs before an interruption, then six continuation epochs from its own saved weights with a fresh optimizer. This is still training from scratch: no externally trained weights were introduced. Continuation uses seed 7, BF16, microbatch 32, accumulation 4, validation batch 32, AdamW learning rate 0.0003, weight decay 0.007, clipping at 1.0, a cosine schedule, dropout 0.12, and word dropout 0.10. The initial stage used learning rate 0.001. Public loss weights are decision 1.0, sensitive 0.2, approval 0.2, and attack 1.0; the private intent objective has weight 1.0.

Selection maximizes validation macro-F1, with validation NLL breaking ties. The selected checkpoint is continuation epoch 5, overall epoch 10 of 11 completed epochs. FP32 validation accuracy is 99.542857%, macro-F1 is 0.9946640065593496, and validation-fitted temperature is 2.7302884615384615. Epoch selection and calibration use validation only. Full training wall time and peak memory were not captured across interruptions: runtime is explicitly null. CPU finalization took 43.69 seconds; that is not training time.

## Resource limits and recovery

The host has approximately 7.6 GiB RAM and an RTX 4050 Laptop GPU with approximately 6 GiB VRAM. The initial larger run coincided with an editor problem; its cause could not be confirmed from available logs. Training was stopped before resuming with safeguards. Continuation used a 2 GiB systemd RAM cap, no worker swap allocation, low scheduling priority, a 150% CPU quota, and a 20% PyTorch GPU allocator limit. It stopped before another epoch when the host-memory check found insufficient available RAM.

Training features are stored as compact, memory-mapped arrays and converted to integer tensors per batch. Validation uses small batches. Selected checkpoints are written atomically. CPU finalization calibrated the saved validation-selected checkpoint under a 1 GiB worker RAM cap. No training job is needed to use the completed artifacts.

## Historical guard and development feedback

The original 21,306-parameter checkpoint remains unchanged at `artifacts/guard_best/model.pt`, SHA-256 `1cce044dff5a31dd143a358cecf36e97552ed6bff5dbbe241941af90d1990551`. Legacy checkpoints load through the same API. Artifact tests check its hash, state keys, and frozen predictions. The original six-epoch CPU training trace and baseline comparison summaries remain in the audit.

Its manual score was 13/24, including 0/10 unfamiliar attack phrasings. Inspected errors motivated architecture and augmentation changes, so the manual probes are a development benchmark. Seven smaller candidates and two earlier wording checks (224 and 128 examples) are retained as development evidence after their results were inspected. The larger candidate is the eighth recorded architecture/data trial.

## Frozen SmolLM2 adapter

Backbone: `HuggingFaceTB/SmolLM2-135M-Instruct`, revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`, with 134,515,008 base parameters. Rank-8, alpha-16, dropout-0 LoRA targets all linear layers, with 2,442,240 trained adapter parameters and frozen base weights.

The completed run uses its original 10,000 training and 1,500 validation examples, seed 42, three epochs, BF16, microbatch 8, accumulation 4, AdamW learning rate 0.0003, weight decay 0.01, 10% warmup, and clipping at 1.0. Loss covers four answer tokens and EOS, excluding prompt and padding. Validation selects epoch 3; the adapter is merged into FP32 and validation-calibrated at temperature 0.45.

Training took 1,549.23 seconds (25.82 minutes), with 3,212.17 MiB peak allocated and 3,676 MiB peak reserved GPU memory. Its [training record](../experiments/smollm2_finetuning.json) supplies the curves. This adapter was not retrained during the Guard refactor. Task-data and training-compute budgets differ between the models.

## Final wording check

The final `generalization` bank was frozen before larger-model training: **256 examples across 64 context strings**, each evaluated under four typed states. Its canonical sorted-row SHA-256 is `e1a09d84786edf573f1469b62d52f94ffd3665afae256a948dc69451ddca998a`; a regression test fixes that snapshot. It was excluded from gradient updates, epoch selection, and calibration.

Guard scores 232/256 (90.625%); LoRA scores 174/256 (67.96875%). This overall result hides a material tradeoff: Guard handles all 96 benign and 64 ordinary-context cases but misses 24/72 untrusted attack cases. LoRA detects all 72 attacks while making many benign and ordinary-context errors. The predeclared 95% overall target remains unmet.

This authored bank has no exact context overlap with the new training/validation banks. It shares concepts and vocabulary and is not an independent real-world security benchmark. Repeated states within each context are correlated; example-level bootstrap intervals do not account for that clustering.

## Matched evaluation

The [complete comparison](FINETUNED_COMPARISON.md) derives from four `experiments/finetuned_*.json` reports. Every quality example is retained, including errors, auxiliary outputs, probabilities, generated text, and null or measured latency. Metadata records checkpoint/adapter hashes, versions, temperatures, input hashes, and timed IDs.

Both models run on the same RTX 4050 Laptop GPU in FP32 with two host threads and TF32 disabled. Both remain loaded during timing; five warmup examples precede alternating measurement order on identical IDs. Timing includes tokenization, all four outputs, decoding, and probability extraction, with CUDA synchronization. Loading, merging and warmup are excluded. Workers had a 1.5 GiB host-RAM cap, no swap allocation, low priority, and a 150% CPU quota. These constrained reference-implementation measurements depend on desktop load and hardware.

Quality covers 1,500 test, 600 challenge, 24 manual, and 256 final-wording examples. Batch-one timings cover 100/100/24/100 matching IDs. The first three sets reuse previously executed frozen LoRA quality only after verifying adapter identity, input hashes, prompt, temperatures, IDs and gold labels. All reported latencies are new; timed outputs are individually rechecked. Both models' final wording predictions are newly executed. LoRA quality uses batch 16 in cached runs and batch 4 for the new bank; throughput is not request latency. p95 uses linear interpolation. Free generation is a separate diagnostic.

## Reproduce

```bash
make setup
make data                     # only if original data is absent
make setup-gpu
# Train a fresh candidate only when desired:
make improve-guard
# Reuse the completed adapter, or train it once:
make finetune
make benchmark ARGS='--split test'
make benchmark ARGS='--split challenge'
make benchmark ARGS='--split manual_probes'
make benchmark ARGS='--split generalization'
make plots report
make check test-artifacts
make verify-package
```

These commands train a fresh candidate; they do not reproduce an interrupted optimizer trajectory byte for byte. To continue only the saved Guard weights with recorded continuation settings, use `make improve-guard ARGS='--resume-from artifacts/guard_scaled/model.pt --lr 0.0003 --epochs 8'`. The optimizer is fresh. After a resource stop, `python -m systemone_guard.robust_train --out artifacts/guard_scaled --finalize-log /path/to/captured-run.log` validates the selected epoch and calibrates on CPU. Recalibration rewrites the checkpoint, so regenerate measurements after changing it.

Override `GUARD_CHECKPOINT=artifacts/guard_best/model.pt` for historical comparison. [The evaluation protocol](EVALUATION.md) defines accuracy effects, timing denominators, and interpretation limits.
