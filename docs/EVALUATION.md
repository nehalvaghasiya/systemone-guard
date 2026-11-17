# Evaluation protocol

The published comparison is **SystemOne Guard (756,248 parameters)** versus **SmolLM2-135M-Instruct adapted with rank-8 LoRA (134,515,008 merged parameters)**. The comparison loader requires a completed adapter run, matching the pinned model revision; it cannot silently substitute an unadapted model.

The updated guard uses 90,000 training and 3,500 validation examples, including compositional context augmentation. The frozen SmolLM2 adapter retains its original 10,000 training and 1,500 validation examples. These task-data budgets differ. Selection and decision-temperature fitting use validation only. Evaluation covers 1,500 test examples, 600 synthetic challenge examples, the 24 previously inspected diagnostic probes, and a new frozen set of 256 examples with 64 separately worded contexts across four typed states. The existing probes are now a development benchmark because their errors motivated the redesign. The new wording set was frozen before candidate training. Neither model trains or calibrates on those sets.

The API supplies the request, untrusted context, and typed action/sensitivity/destination/authorization/source fields. The updated guard uses context and typed state; request wording is irrelevant to the simulated policy and is excluded from its encoder. SmolLM2 receives all public fields and its policy prompt. Gold decisions, gold auxiliary values, and attack-template annotations are excluded from inference inputs. SmolLM2 also receives an explicit policy prompt. Guard produces four heads simultaneously; SmolLM2 decodes decision, sensitive, approval-needed, and attack words. Its adapter is merged before evaluation.

## Quality

Every example contributes to accuracy, macro-F1, per-class precision/recall/F1, confusion matrices, multiclass NLL, sum-over-classes Brier score, 10-bin ECE, and all three auxiliary accuracies. Decision probabilities are calibrated on validation. SmolLM2 probabilities are normalized over the allowed decision words; its binary probabilities condition on preceding decoded words. Guard's heads are simultaneous.

SmolLM2 quality inference uses batches of 16 for speed. The common timed subset is rechecked at batch size 1, and those single-request outputs replace the corresponding quality records. No batched throughput value is labeled as request latency. Invalid predictions count as incorrect. An additional greedy free-generation diagnostic requires exactly four valid words, with a maximum of eight tokens; its quality and latency are reported separately.

Accuracy gain always identifies its direction:

- **LoRA − Guard absolute gain (pp)** = `100 × (LoRA correct − Guard correct) / quality n`.
- **LoRA relative accuracy gain (%)** = `100 × (LoRA correct − Guard correct) / Guard correct`. This is undefined if Guard has zero correct examples.
- **Additional correct examples** = `LoRA correct − Guard correct`, with negative values preserved.

Paired accuracy intervals resample the per-example correctness differences 10,000 times with seed 42. This is equivalent to resampling matched example pairs. Intervals reflect uncertainty within these fixed datasets; there are no repeated final training seeds. The new wording set shares context strings across four typed states, so example-level intervals do not account for context clustering. The manually selected probes are diagnostic and do not estimate a deployment population.

## Latency

Both models run on the same NVIDIA RTX 4050 Laptop GPU in FP32, with two host threads and TF32 disabled. Both remain loaded during timing. After five warmup examples per model, model order alternates across matched requests. CUDA synchronizes before and after each measurement.

The timed boundary includes Python tokenization, host/device transfer, model execution, all four outputs, decoding, and probability extraction. It excludes loading, adapter merging, and warmup. One hundred seed-42 shuffled IDs are timed on each synthetic split; all 24 manual probes and 100 new-wording examples are timed. The same IDs are used by both models. Other records have null latency.

Report median, p95, and mean separately. p95 is `numpy.quantile(times, 0.95, method="linear")`. Each effect uses the same statistic in both models:

- **Guard saved latency (ms)** = `LoRA ms − Guard ms`.
- **Guard latency reduction (%)** = `100 × (LoRA ms − Guard ms) / LoRA ms`.
- **Guard speedup (×)** = `LoRA ms / Guard ms`.

Speedup and percentage reduction are different quantities. A ratio of model medians is not a median of per-example ratios. These are end-to-end measurements of this implementation and decoding scheme on one laptop, with a limited tail sample; machine load, thermal state, library versions, and optimized runtimes can change them.

## Evidence and regeneration

[The complete report](FINETUNED_COMPARISON.md) and all six figures come from `experiments/finetuned_{test,challenge,manual_probes,generalization}.json`. Each report retains matched IDs, raw probabilities, auxiliary outputs, measured/null latencies, free-generation records, model hashes, data hashes, hardware, versions, temperatures, and exact effect formulas.

`make report` and `make plots` recompute summaries and effects from those records. They reject extra models, mismatched IDs, inconsistent gold labels, different timing subsets, nonpositive latency, and stale saved summary values. All published comparisons use this validated pair.

```bash
make benchmark ARGS='--split test'
make benchmark ARGS='--split challenge'
make benchmark ARGS='--split manual_probes'
make benchmark ARGS='--split generalization'
make plots
make report
```

`make evaluate` is a guard-only diagnostic that writes to ignored `artifacts/guard_evaluation`; it does not supply published comparison tables. Evaluation inputs and hardware are matched, while task-data budgets, architecture, prior training, input representations, parameter budgets, and training compute differ.
