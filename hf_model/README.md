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
- system-one-inspired
---

# SystemOne Guard (21K) — Typed Agent Tool Decisions

## Summary

**Model:** A small, original parallel probabilistic decision classifier trained from random initialization on synthetic policy-labeled agent tool calls. **Not Jev** (TypeSafe's architecture and RLCD recipe are private), not a pretrained LLM, and **not security-grade**.

**Input:** Agent request, untrusted tool result or retrieved text, and a typed state consisting of action, sensitivity, destination, authorization, and source trust.

**Output:** Three-choice `allow/review/block` probabilities plus `sensitive`, `approval_needed`, and `attack` auxiliary probabilities computed in one forward pass. Output is a fixed schema, not generated text.

## Architecture

Custom word embeddings, parallel depthwise 3-gram and 5-gram CNN filters, token mixing, attention pooling, typed categorical embeddings, fused hidden layer, and four heads. Selected checkpoint contains **21,306 total parameters** and **no pretrained weights**.

## Training

- Random initialization, PyTorch, supervised multi-task cross entropy (decision weight 1.0; sensitive 0.2; approval 0.2; attack 0.35).
- Data: 10,000 generated training examples, 1,500 validation, 1,500 historical test, 600 synthetic challenge.
- Selected run: seed 7, dim 32, hidden 64, AdamW LR 0.0015, batch 128, six epochs, 2 CPU threads.
- Selection: validation macro-F1. Temperature calibration: temperature 0.45 selected on validation log loss (grid search).
- Reproducibility: source and JSONL files in the accompanying GitHub/ZIP project. All weights, run histories, per-example predictions, and graphs supplied.

## Evaluation

| Method | Historical test accuracy | Challenge accuracy |
|---|---:|---:|
| SystemOne Guard | **99.87%** | **100%** |
| Word TF-IDF + logistic | 90.67% | 91.50% |
| Character TF-IDF + logistic (4K training subset) | 81.40% | 80.83% |
| Scratch causal next-label transformer (10,000 examples) | 88.13% | 78.83% |

**Diagnostic manual probes: 13/24 (54.17%) correct; zero of 10 previously unseen adversarial attack phrasings detected.** This is a striking distribution-shift failure. The policy generator is deterministic, and a faithful hand-written rule engine can be perfect on the generated task. The synthetic benchmark does **not** demonstrate production prompt-injection security. Report these failures with the headline scores.

For exact probabilities, metrics, confusion matrix, calibration, and predictions, see `experiments/` in the complete project.

## Intended use

Research and education in typed decision models, efficiency/latency comparison and calibration. It can be used to build an illustrative interactive demonstration of single-forward-pass classification.

**Not intended for:** Production security gates, permissions enforcement, safety-critical workflows, fraud decisions, moderation at scale, or situations where incorrectly allowing an operation creates harm.

## Usage

```python
from systemone_guard.inference import DecisionEngine
engine = DecisionEngine('artifacts/guard_best/model.pt')
result = engine.decide(
    request='Read team documents for an audit',
    context='Tool returned a normal response.',
    action='read', sensitivity='internal', destination='internal',
    authorization='approved', source='user',
)
print(result)
```

You must have the custom model source and trained checkpoint; this is not automatically loadable via `AutoModelForSequenceClassification` without adapting the config. Model weights are stored with the project as `artifacts/guard_best/model.pt`.

## Responsible use and failure analysis

Dataset is synthetic. Model learns the simulated policy and familiar attack-template words rather than genuine robust detection. It can output strongly confident *incorrect* answers on new attack types. Always separate deterministic permission checks from probabilistic model suggestions and require human oversight for consequential operations.

## Provenance / license

Original custom PyTorch model, original synthetic dataset, MIT licensed demo source. Jev name and technology are TypeSafe AI's, and this checkpoint is not associated with TypeSafe.
