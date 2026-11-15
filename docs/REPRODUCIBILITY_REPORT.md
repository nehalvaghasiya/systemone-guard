# Reproducibility and experiment ledger

Project date: 2026-11-07. Environment: PyTorch 2.10.0 CPU, scikit-learn 1.8.0, NumPy 2.3.5, two CPU threads for training and latency, Python 3.13.

## Training runs

| Run | Architecture | Seed | Widths | Epochs | Training examples | Validation macro-F1 | Saved weights |
|---|---|---:|---|---:|---:|---:|---|
| run_d16 | Parallel decision network | 99 | embed16/hidden32 | 5 | 10,000 | 1.000 | Yes |
| run_d32 **selected** | Parallel decision network | 7 | embed32/hidden64 | 6 | 10,000 | 1.000 | Yes |
| run_d48 | Parallel decision network | 42 | embed48/hidden96 | 6 | 10,000 | 1.000 | Yes |
| run_d64 | Parallel decision network | 99 | embed64/hidden128 | 1 (interrupted) | 10,000 | 0.968 | Partial checkpoint only |
| causal decoder | Tiny causal Transformer, next-label | 42 | dim32/1 layer | 4 | 10,000 | 0.702 | Yes |
| Word TF-IDF | Logistic regression | 42 | word 1–2 grams | n/a | 10,000 | See metrics | Yes |
| Char TF-IDF | Logistic regression | 42 | char 2–5 grams | n/a | 4,000 | See metrics | Yes |

Selection used validation macro-F1 and smaller capacity preference for ties (the run_d16 experiment was completed after run_d32 selection and after test data were seen; **do not reselect using d16 as a post-test improvement**). Temperature calibration uses validation-only negative log likelihood. The archived model is the `run_d32` checkpoint.

## Test metrics

- Selected SystemOne Guard: **99.87% accuracy**, **99.65% macro-F1**, 10-bin ECE **0.00275**.
- Word TF-IDF: **90.67% accuracy**, **87.75% macro-F1**.
- Character TF-IDF: **81.40% accuracy**, **64.73% macro-F1** (4K training examples).
- Causal decoder from scratch: **88.13% accuracy**, **73.29% macro-F1** (10K training examples).
- Test class proportions: 104 allow, 312 review, 1,084 block. Most decisions are blocks.

Paired bootstrap test difference against word TF-IDF: **+9.20 percentage points** (95% interval **+7.80 to +10.73 pp**). These confidence intervals describe finite-sample uncertainty on the synthetic test; they **do not account for dataset generation bias or evaluation leakage**. Comparisons for other baselines in `experiments/bootstrap_summary.json`.

## Critical external validity failure

The manual 24-case diagnostic scored **54.17% overall**: 0/10 novel malicious phrasing, 5/5 benign quotation examples, 8/9 policy permutations. The model memorizes familiar adversarial lexicon from the generator. Even high-confidence predictions should be treated as untrustworthy on novel malicious instructions. Never use as a production enforcement model.

## Real pretrained LLM benchmark not completed

The runnable external comparator is `HuggingFaceTB/SmolLM2-135M-Instruct` (Apache-2.0, downloaded by `systemone_guard.benchmarking.pretrained`), which cannot be retrieved in the local network-restricted environment. Run the comparison in Google Colab with a fresh connected runtime; until then, there is no measured head-to-head claim against a pretrained System Two model.
