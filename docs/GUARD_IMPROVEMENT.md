# Guard generalization experiment

Goal: improve unfamiliar-context decisions while preserving a small, fast, single-pass learned classifier. The user-visible contract remains decision probabilities and three auxiliary risk probabilities.

Baseline: the original 21,306-parameter guard gets 13/24 manual probes correct, including 0/10 novel attacks; the frozen SmolLM2 LoRA comparator gets 23/24 and 9/10. The existing probes have already been inspected and are now a development benchmark, not a new blind test.

Hypothesis: eleven fixed training contexts and a closed word vocabulary encourage template memorization. Diverse labeled instructions, matched benign quotations, counterfactual typed states, and character-subword features should reduce this failure without introducing a language-model dependency at inference.

Data contract: retain the original data and selected checkpoints. Generate additional training/validation examples from separate wording banks. Freeze a separate generalization set before training. Never train on the manual probes, original test/challenge rows, or the generalization holdout. Store hashes, exact training configuration, source family counts, model size, and measured results. Check for duplicate context text between the new wording banks and manual probes. Original synthetic contexts already overlap across the original splits by design.

Selection: use the combined original/new validation set only, maximizing decision macro-F1 with lower NLL breaking ties. Fit temperature using that validation set only. Manual-probe or final-holdout scores must not select epochs, thresholds, seeds, or checkpoints.

Targets: at least 23/24 existing probes correct, at least 99.5% on each original synthetic split, at least 95% on a fresh holdout with benign quotations and unfamiliar attack wording, and batch-one median latency below one tenth of SmolLM2 LoRA's on the same device. Report attack recall, benign false positives, calibration and all auxiliary heads as well as overall accuracy. These are experiment acceptance targets, not guarantees of security or broad model equivalence.

Comparison scope: the new guard receives broader task training data. The existing SmolLM2 adapter stays frozen; disclose the changed training budget rather than claiming identical task data. Evaluate both models on the same final inputs and matched timing protocol.

Constraints: use the available local GPU, retain checkpoint compatibility, keep preprocessing identical in training and inference, and compute predictions from public inputs only. No lookup of probe IDs/text or gold simulator labels is allowed in inference. Prior art for subword representations: [Bojanowski et al.](https://arxiv.org/abs/1607.04606).

Rollback: keep the original checkpoint at `artifacts/guard_best/model.pt`. The selected larger candidate lives under `artifacts/guard_scaled/`. Existing staging remains untouched.

## Recorded experiment sequence

Seven smaller candidates explored broader context data, subwords, pooling, attention, and contrastive examples. The sixth reached 19/24 manual probes and 201/224 on the first wording check; the seventh reached 17/24 and 116/128 on a second wording check. Those inspected wording results are now development evidence. Their snapshots and failed outcomes remain in the experiment audit.

The user then required every Guard weight to remain trained from scratch and allowed a larger model and more training. Before that larger run, a final 256-example bank across 64 contexts was frozen, excluded from training/selection, and fixed by a hash regression test. The eighth candidate has 756,248 parameters, three attention layers, a private source-independent context-intent objective, and learned typed-state policy heads. It uses 90,000 training and 3,500 validation examples; the comparator retains its original 10,000/1,500 task data.

The larger run completed five initial epochs and six continuation epochs from its own saved weights, with a fresh optimizer. Validation selected overall epoch 10. A resource safeguard stopped continuation before another epoch; CPU finalization checked the saved validation score and fitted temperature. Full training runtime was not captured and is not estimated from smaller runs. Feature arrays are memory-mapped, batches are small, and Linux training scopes cap worker RAM at 2 GiB with no swap allocation and a 150% CPU quota. The original checkpoint remains unchanged. See [the reproducibility record](REPRODUCIBILITY_REPORT.md) for settings and recovery commands.

The final manual target is met, but the final wording target remains unmet. Higher overall wording accuracy also hides weaker attack recall: Guard misses 24/72 untrusted attack cases, while LoRA detects all 72 and makes more benign-context errors. This result supports the reported development improvement, not universal parity with LoRA or real-world security claims.

<!-- BEGIN IMPROVEMENT RESULTS -->

## Measured outcome

| Set | Original guard correct | Updated guard correct | Frozen SmolLM2 LoRA correct |
|---|---:|---:|---:|
| Test | 1498/1500 | 1500/1500 | 1500/1500 |
| Challenge | 600/600 | 600/600 | 599/600 |
| Manual probes | 13/24 | 24/24 | 23/24 |
| New wording | not measured | 232/256 | 174/256 |

| Predeclared target | Result |
|---|---|
| manual at least frozen smollm | PASS |
| test accuracy at least 99 5 percent | PASS |
| challenge accuracy at least 99 5 percent | PASS |
| new wording accuracy at least 95 percent | FAIL |
| median latency below one tenth of smollm | PASS |

Updated Guard has 756,248 parameters. The original 21,306-parameter checkpoint remains loadable and unchanged. The updated model receives broader task training data; the SmolLM2 adapter remains frozen.

The fresh wording check is 232/256 for Guard and 174/256 for LoRA. Its separately authored contexts are repeated across four typed states and do not establish general language-model equivalence.

See [the complete comparison](FINETUNED_COMPARISON.md) for quality, calibration, auxiliary outputs, incorrect IDs, matched latency, and data-budget qualifications.

<!-- END IMPROVEMENT RESULTS -->
