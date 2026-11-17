---
language:
- en
license: mit
pipeline_tag: text-classification
tags:
- from-scratch
- multi-task
- structured-decisions
- calibrated-classification
- synthetic-data
---

# SystemOne Guard (756K)

A custom 756,248-parameter PyTorch classifier trained from random initialization on synthetic agent tool-call policy labels. Inputs are request text, untrusted context, action, sensitivity, destination, authorization, and source. Outputs are decision probabilities plus sensitive-data, approval, and attack probabilities in one forward pass.

The updated network combines word/character-subword embeddings, parallel convolutions, three context attention layers, attention/mean pooling, and a private context-intent head. Learned policy heads use intent and typed state to return four public outputs. It preserves the public request argument; all Guard weights are trained from scratch.

Guard uses 90,000 training examples and 3,500 validation examples, including compositional attack/benign augmentation. The existing SmolLM2 LoRA adapter remains frozen with its original 10,000/1,500 task splits. The task-data budgets differ. Guard checkpoint selection and temperature fitting use validation only. The prior 21K checkpoint is preserved for compatibility and historical comparison.

## Measured fine-tuned comparison

<!-- BEGIN MEASURED COMPARISON -->

Measured comparison with **SmolLM2-135M-Instruct fine-tuned using LoRA** (134,515,008 merged parameters) and **SystemOne Guard** (756,248 parameters). Both return the decision and all three auxiliary outputs.

| Split | Guard correct / total | LoRA correct / total | Guard accuracy | LoRA accuracy | LoRA - Guard |
|---|---:|---:|---:|---:|---:|
| Test | 1500/1500 | 1500/1500 | 100.0000% | 100.0000% | +0.0000 pp (+0 correct) |
| Challenge | 600/600 | 599/600 | 100.0000% | 99.8333% | -0.1667 pp (-1 correct) |
| Manual probes | 24/24 | 23/24 | 100.0000% | 95.8333% | -4.1667 pp (-1 correct) |
| New wording | 232/256 | 174/256 | 90.6250% | 67.9688% | -22.6562 pp (-58 correct) |

Accuracy differences above are **percentage points**, with Guard as the accuracy baseline. Positive differences favor LoRA; negative differences favor Guard. The previously inspected manual probes are a development benchmark. The final wording set has 256 examples from 64 separately authored contexts and was frozen before candidate training; it is still a synthetic diagnostic.

| Split | Timed requests per model | Guard median | LoRA median | Guard saves | Guard latency reduction | Guard speedup |
|---|---:|---:|---:|---:|---:|---:|
| Test | 100 | 4.109 ms | 112.031 ms | 107.922 ms | 96.33% | 27.27x |
| Challenge | 100 | 3.370 ms | 109.902 ms | 106.531 ms | 96.93% | 32.61x |
| Manual probes | 24 | 3.290 ms | 101.153 ms | 97.863 ms | 96.75% | 30.75x |
| New wording | 100 | 3.438 ms | 114.772 ms | 111.335 ms | 97.00% | 33.39x |

Latency is measured on the **same NVIDIA GeForce RTX 4050 Laptop GPU**, FP32, batch size 1, 2 host threads, with device synchronization and tokenization, all four outputs, decoding and probability extraction included. Both models remain loaded; measurement order alternates per example after five warmup examples each. Loading, adapter merging, and warmup are excluded. Quality uses every example; untimed records have null latency. Quality batches are not reported as request latency.

**Latency reduction** = `100 x (LoRA ms - Guard ms) / LoRA ms`. **Speedup** = `LoRA ms / Guard ms`. Each ratio uses the same statistic (median, p95, or mean); reduction is not a speedup percentage.

The recorded inference workers used a 1.5 GiB host-RAM cap and a 150% CPU quota to preserve desktop responsiveness. These are measurements of this constrained reference implementation. Previously executed LoRA quality may be reused only when model, input and prompt hashes match; every reported latency is freshly measured.

<!-- END MEASURED COMPARISON -->

The [complete comparison](https://github.com/nehalvaghasiya/systemone-guard/blob/main/docs/FINETUNED_COMPARISON.md) includes calibration, auxiliary heads, per-class metrics, free generation, accuracy uncertainty, errors, and all six paired figures. Source records are `experiments/finetuned_*.json`; actual training traces are `experiments/guard_training.json` and `experiments/smollm2_finetuning.json` in the repository.

## Usage

```python
from systemone_guard.inference import DecisionEngine

engine = DecisionEngine("artifacts/guard_scaled/model.pt")
result = engine.decide(
    request="Read team documents for an audit",
    context="Tool returned a normal response.",
    action="read",
    sensitivity="internal",
    destination="internal",
    authorization="approved",
    source="user",
)
print(result)
```

The custom `systemone_guard` source and checkpoint are required. These weights are not a standard Transformers sequence classifier. The repository's Gradio interface is a demonstration and does not execute tools.

## Scope and failures

Use for research and education about typed decisions, calibration, and inference cost. Labels come from a simulated policy; synthetic scores do not establish real-world security performance. Manual accuracy improved from 13/24 to 24/24, versus LoRA's 23/24. On the separate frozen 256-example bank, Guard scores 232/256 but misses 24/72 untrusted attacks; LoRA detects all 72 while making more benign-context errors. The 95% overall wording target remains unmet. Probabilistic suggestions must remain separate from deterministic permissions enforcement.

Original model code and synthetic data are MIT licensed. This Jev-inspired project is independent of TypeSafe AI and does not reproduce its proprietary system. SmolLM2 provenance, pinned revision, adapter configuration, and data hashes are recorded in the accompanying training report.
