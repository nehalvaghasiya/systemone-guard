# Evaluation methodology and caveats

Generated data is reproducible from seed 20261107. The label teacher is `policy_decision` in `src/systemone_guard/data.py`.

- Training: 10,000 policy-synthetic examples, 14 user-request templates.
- Validation: 1,500 examples from the same request-template pool, independent random seed.
- Test: 1,500 examples using three held-out request templates.
- Challenge: 600 examples using three additional held-out templates plus a shifted action/attack distribution.
- Manual probes: 24 manually authored examples, ten adversarial attacks using unseen phrasing, five benign quotations, nine policy permutations. **These probes were not used during training or checkpoint selection.**

**Important split limitation:** Although request templates are held out, attack context templates, policy function, object vocabulary and most semantics are shared. Thus test/challenge scores are optimistic and cannot support claims of general language comprehension or security. The manual probes provide evidence of severe out-of-distribution failure.

Model selection by validation macro-F1 and smaller parameter count on ties. Grid-search temperature 0.45–3.0 on validation negative log likelihood only. Test and probes are never used to select the winning checkpoint. Results were observed during development, so do not describe the test as newly blind.

Reported: accuracy, macro-F1, NLL, multiclass Brier, 10-bin ECE, confusion matrix, auxiliary accuracy, batch inference latency. Word TF-IDF is full-data 10K; char TF-IDF trained on only 4K; causal decoder trained on 10,000 examples for four epochs with the same structured state serialized into its prompt. The causal comparator is **not** a pretrained instruction LLM, therefore the report does not establish superiority over pretrained language models.

Real pretrained comparison: run `systemone_guard.benchmarking.pretrained` using `HuggingFaceTB/SmolLM2-135M-Instruct` in a networked environment on the exact same records. Record parsing failures and end-to-end CPU timings for both methods on the same seeded subset. This baseline was not executed locally due missing internet/model weights.

Recommended follow-up: collect real consented and safely de-identified agent traces, explicit attack-vs-quotation labels, multi-source task contexts, independent human annotations, novel adversarial phrasing, robust calibration and downstream security-violation rate. Use a fully independent test set before any hyperparameter adaptation.
