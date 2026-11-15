# Development workflow

Use GNU Make and uv from the repository root. Python 3.11 is the default;
Python 3.10 remains supported and is tested in CI.

```bash
make setup
make format
make check
make build
make help
```

`make setup` installs the exact versions in `uv.lock`, including CPU PyTorch
and the development extra. Every tool command uses that same locked environment.
Use `make lock` after changing dependency declarations; use
`make lock ARGS=--upgrade` only for an intentional dependency upgrade.
The CPU index is explicit and applies only to PyTorch.

The standard pip editable installation remains available for Colab and consumers:
`python -m pip install -e '.[dev]'`. Pip does not use the uv lockfile.
For Colab, install the package with its notebook extra:
`python -m pip install -e '.[dev,notebook]'`.

## Layout

- `src/systemone_guard`: importable model, data, training, metrics, inference,
  checkpoint loading, typed dictionary contracts, and reusable benchmark helpers.
- `scripts`: explicit developer and research commands with `main()` functions.
- `spaces`: optional Gradio adapter with a `create_demo()` factory.
- `tests/unit` and `tests/integration`: deterministic tests and workflows.
- `notebooks`, `docs`, and `hf_model`: notebooks and research documentation.
- `data` and `artifacts`: ignored local datasets and trained weights.
- `experiments` and `figures`: preserved published research outputs.
- `dist`: generated Python distributions and local Space exports.

Install the project before importing it. The source layout intentionally prevents
accidental imports of an uninstalled package from the checkout.

## Checks and conventions

Ruff handles formatting, import sorting, and linting. Python uses four spaces,
double quotes, and an 88-character formatting width. Make recipes use tabs.
Ruff also checks notebook code cells; interactive notebook imports have narrowly
scoped exceptions for unused imports and installation-before-import cells.

For Python files, `make format` also inserts one blank line around control-flow
sections, before returns after computations, and when typed declarations start
a new section. Adjacent related assignments stay together. Existing section
breaks, attached comments, multiline strings, and `# fmt: off/on` regions are
preserved. Notebook cells use Ruff's native formatting.

Ruff is allowed to fix import ordering only. Lint rules that would rewrite
expressions, variable names, conditions, or return values remain diagnostic.
The section-spacing pass compares Python ASTs before writing each file.

Strict mypy checks every handwritten Python file, including scripts, tests,
and the demo. Shared TypedDicts retain existing dictionary and JSON shapes.
Optional or untyped integrations have module-specific import exceptions.
nbformat's validation APIs are the only library-wide untyped-call exception.
No project modules are excluded from type checking.

`make check` checks formatting, lint, typing, and the default test suite.
`make format` applies import sorting, wrapping, indentation, and section spacing.
`make format-check` verifies all of them without changing files. Optional local
hooks can be installed with `make hooks`; they use the same formatting convention.
`make coverage` reports branch coverage without imposing a percentage gate.

## Tests and packaging

Default tests generate small datasets and checkpoints in temporary directories.
They require neither local research files, GPUs, model downloads, nor credentials.
Packaging and optional-adapter tests have explicit pytest markers and run through
separate Make targets; the default suite excludes them.

```bash
make test
make test ARGS='-k checkpoint'
make test-artifacts
make verify-package
make audit
```

The artifact target explicitly requires the original local checkpoint and dataset
splits. It checks the checkpoint hash, state keys, output schema, and saved
probabilities against the pre-refactor snapshot with an absolute tolerance of
`1e-6`. Missing assets fail with the required path.

Packaging builds both a wheel and source distribution and checks their metadata.
`make verify-package` builds once, then runs the packaging integration test. It
installs the wheel into a temporary environment using the locked CPU dependencies
and verifies imports and inference outside the checkout.
`make audit` accesses online vulnerability advisories and skips the local
project. It maps CPU wheel suffixes such as `+cpu` to the upstream PyTorch
release so the dependency is audited. CI runs it as a separate job.

CI runs `make check verify-package` on Python 3.10 and 3.11. Optional-adapter
integration tests run through `make smoke-extras` with the corresponding extras.
The adapter test constructs the demo with a temporary checkpoint; it neither
downloads pretrained weights nor uploads files.

## Research commands

```bash
make data ARGS='--out data --seed 20261107'
make train ARGS='--data data --out artifacts/reproduced --epochs 6 --seed 7 --dim 32 --hidden 64 --lr 0.0015 --batch 128'
make evaluate ARGS='--split test --out experiments/new-run'
make demo
make benchmark ARGS='--split test --count 100'
make prepare-space ARGS='--out dist/my-space'
```

Research commands can write results to their requested locations. Choose separate
output paths when preserving published results. The matched pretrained benchmark
is the single comparison implementation:
`python -m systemone_guard.benchmarking.pretrained`. It uses a seeded subset,
strict output parsing, and end-to-end CPU timing for both methods.

The notebooks are maintained directly and validated by the test suite. Plotting
uses `scripts/plots.py results` or `scripts/plots.py pretrained --input ...`.
Space preparation uses `scripts/huggingface.py prepare`; explicit model uploads
use `scripts/huggingface.py publish --repo ...` with the publish extra and HF_TOKEN.

`make demo` and `make benchmark` enable the corresponding optional dependencies.
For notebook work, run `uv sync --locked --extra dev --extra cpu --extra notebook`.
For a GPU environment, omit the CPU extra, for example:
`make setup EXTRAS='--extra dev --extra llm'`. Pass the same `EXTRAS` override to
subsequent Make commands. The matched pretrained benchmark retains its CPU-only
comparison policy.

`make clean` removes Python caches, coverage outputs, and generated distributions.
It preserves source, notebooks, datasets, checkpoints, experiments, and figures.
