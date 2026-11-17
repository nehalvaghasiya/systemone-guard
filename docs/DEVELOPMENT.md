# Development

Use Make as the command interface. `make help` lists commands and `ARGS='...'` passes options to a workflow. Python 3.10+ is supported for CPU development; the recorded GPU run uses Python 3.11.

```bash
make setup
make format
make check
make test-artifacts
make verify-package
```

`make format` sorts imports with Ruff, applies Ruff formatting, inserts blank lines between logical sections, and formats again. `make format-check` checks the same convention without editing. `make lint` runs Ruff and `make typecheck` runs strict mypy across source, scripts, the demo, and tests. Formatting keeps model logic and expressions intact.

The self-contained tests use small generated datasets and temporary checkpoints. They test training, checkpoint compatibility, inference, matched evaluation, calibration, effect denominators, timing alignment, report validation, plots, notebooks, exports, and import safety. Artifact tests additionally check the original local checkpoint and data. Packaging tests build and install the wheel in isolation. `make smoke-extras` checks optional integrations without downloading model weights or publishing.

## Research workflows

```bash
make data
make setup-gpu
make improve-guard
make evaluate ARGS='--split test'

make setup-gpu
make finetune
make benchmark ARGS='--split test'
make benchmark ARGS='--split challenge'
make benchmark ARGS='--split manual_probes'
make benchmark ARGS='--split generalization'
make plots
make report
```

`make setup-gpu` creates `.venv-gpu` using the locked CUDA 12.6 PyTorch distribution and LoRA dependencies, leaving the CPU environment separate. `make benchmark` requires `artifacts/smollm2_lora/best`; override it with `ADAPTER=...`. A complete run and matching revision are mandatory. The guard defaults to `artifacts/guard_scaled/model.pt`; override it with `GUARD_CHECKPOINT=...`. `make improve-guard` builds disjoint context banks, checks exclusion of manual-probe text, and selects the subword/attention model using validation only. `make train` retains the legacy CPU training utility. Explicit `ARGS='--device cpu --split test'` supports diagnostic CPU execution, but changes the measurement conditions and requires regenerating the accompanying documentation.

Data and weights are local ignored research artifacts. The compact evidence in `experiments/` consists of four matched pair reports, one training trace per model, and the before/after experiment record. `make plots` creates the six canonical figures. `make report` regenerates the complete comparison and marked overview sections in the README and model card. Both commands validate raw records against saved summaries, so regenerate evaluation reports before updating published results.

## Layout

- `src/systemone_guard/`: data generator, model, training, checkpoints, inference, metrics, and guard-only diagnostics.
- `src/systemone_guard/benchmarking/`: LoRA training, paired evaluation, shared contracts, manual probes, effect calculation, and report generation.
- `scripts/`: formatting, dependency audit, plots, and Hugging Face export.
- `tests/`: CPU unit/integration tests, optional adapters, original artifacts, and distribution checks.
- `notebooks/`: guard training and matched fine-tuned comparison workflows.
- `experiments/`, `figures/`, `docs/`: compact measurements, canonical charts, and methodology.

`make clean` removes caches, build output, and coverage files while retaining datasets, checkpoints, reports, figures, and notebooks. CI runs source checks and distribution validation on Python 3.10/3.11, plus optional integration checks and dependency auditing.

## Demo and export

Install the demo extra before `make demo`. `make prepare-space` exports a complete local Hugging Face Space, including the package source and guard checkpoint. `scripts/huggingface.py publish --repo ...` is an explicit publication command requiring the publish extra and an authenticated account. Local preparation does not publish anything.

## Training on a machine with limited RAM

The larger from-scratch guard streams uint16 token features from a NumPy disk cache. Only a 32-example microbatch becomes int64 tensors in RAM; accumulation keeps the effective batch at 128. Validation also uses batches of 32. The worker checks for at least 1.5 GiB available host memory and caps the PyTorch CUDA allocator at 20% of device memory.

On Linux with a user systemd session, `make improve-guard` also isolates the worker in a scope with a 2 GiB hard RAM cap, no swap allocation, low CPU priority, and a 150% CPU quota. The cap applies to the training worker rather than the editor. `TRAIN_SCOPE=` explicitly disables the systemd wrapper on other platforms; the in-process memory checks remain. Checkpoints are written atomically. The original checkpoint remains separate; use `ARGS='--out artifacts/new_candidate'` to preserve an already measured candidate when starting another run.

`make benchmark` similarly uses a 1.5 GiB worker RAM cap and a 150% CPU quota. `BENCH_SCOPE=` disables that wrapper on other platforms. Run splits sequentially on this small host. `--reuse-reference /path/to/reports` can reuse executed frozen LoRA quality after verifying model/input/prompt identity; timings are always newly measured. Recovery and the selected interrupted-run trajectory are documented in [the reproducibility record](REPRODUCIBILITY_REPORT.md).
