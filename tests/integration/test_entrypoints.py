"""CLI compatibility, import safety, and generated deliverables."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pytest
from scripts.huggingface import prepare
from scripts.plots import create_pretrained, create_results

from systemone_guard.benchmarking.pretrained import summarize
from systemone_guard.benchmarking.types import BenchmarkRecord, BenchmarkResult
from systemone_guard.inference import DecisionEngine
from systemone_guard.metrics import score
from systemone_guard.types import Example, decision_input

ROOT = Path(__file__).resolve().parents[2]


def test_imports_have_no_runtime_side_effects(tmp_path: Path) -> None:
    modules = [
        f"scripts.{path.stem}"
        for path in (ROOT / "scripts").glob("*.py")
        if path.name != "__init__.py"
    ] + ["spaces.app", "systemone_guard.inference"]
    code = """
import importlib
import json
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
with patch('torch.load', side_effect=AssertionError('checkpoint loaded on import')), patch(
    'socket.socket.connect', side_effect=AssertionError('network attempted on import')
):
    for module in json.loads(sys.argv[2]):
        importlib.import_module(module)
assert 'systemone_guard.train' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code, str(ROOT), json.dumps(modules)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "module",
    ["data", "train", "evaluate", "inference", "causal", "benchmarking.pretrained"],
)
def test_module_help(module: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", f"systemone_guard.{module}", "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "name",
    [
        "audit_dependencies",
        "benchmark_latency",
        "bootstrap_summary",
        "evaluate_probes",
        "format_sections",
        "huggingface",
        "plots",
    ],
)
def test_script_help(name: str, tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / f"{name}.py"), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in result.stdout


def test_inference_cli_outside_checkout(
    checkpoint: Path,
    sample: Example,
    tmp_path: Path,
) -> None:
    inputs = decision_input(sample)
    arguments = [
        argument
        for name, value in inputs.items()
        for argument in (f"--{name}", str(value))
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "systemone_guard.inference",
            "--checkpoint",
            str(checkpoint),
            *arguments,
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == DecisionEngine(checkpoint).decide(**inputs)


def test_notebooks_validate() -> None:
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        nbformat.validate(nbformat.read(path, as_version=4))


def test_space_export_includes_full_package(checkpoint: Path, tmp_path: Path) -> None:
    output = prepare(ROOT, tmp_path / "space", checkpoint)
    assert (output / "systemone_guard/benchmarking/pretrained.py").is_file()
    assert (output / "systemone_guard/py.typed").is_file()
    assert (
        output / "artifacts/guard_best/model.pt"
    ).read_bytes() == checkpoint.read_bytes()
    assert "gradio" in (output / "requirements.txt").read_text()
    assert "sdk: gradio" in (output / "README.md").read_text()
    assert not list(output.rglob("__pycache__"))

    with pytest.raises(FileNotFoundError, match="Checkpoint required"):
        prepare(ROOT, tmp_path / "missing-space", tmp_path / "missing.pt")


def test_clean_preserves_research_files(tmp_path: Path) -> None:
    for name in (
        "data",
        "artifacts",
        "experiments",
        "figures",
        "notebooks",
        "src",
        "scripts",
        "spaces",
        "tests",
    ):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "keep").write_text("research")

    for name in (
        "dist",
        "build",
        ".mypy_cache",
        "src/__pycache__",
        "src/systemone_guard.egg-info",
    ):
        (tmp_path / name).mkdir()

    subprocess.run(
        ["make", "--file", str(ROOT / "Makefile"), "clean"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not (tmp_path / "dist").exists()
    assert not (tmp_path / "src/__pycache__").exists()
    assert not (tmp_path / "src/systemone_guard.egg-info").exists()

    for path in tmp_path.glob("*/keep"):
        assert path.read_text() == "research"


def test_demo_remains_importable() -> None:
    assert callable(importlib.import_module("spaces.app").create_demo)


def test_plot_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib-cache"))
    experiments = tmp_path / "experiments"
    experiments.mkdir()
    metrics = score(np.eye(3), np.arange(3))

    for split in ("val", "test", "challenge"):
        (experiments / f"{split}_metrics.json").write_text(
            json.dumps({"systemone_guard": metrics})
        )

    artifacts = tmp_path / "artifacts"

    for name in ("run_d16", "run_d32", "run_d48"):
        directory = artifacts / name
        directory.mkdir(parents=True)
        (directory / "training.json").write_text(
            json.dumps(
                {
                    "history": [
                        {
                            "epoch": 1,
                            "train_weighted_loss": 0.3,
                            "val_macro_f1": 1.0,
                            "val_nll": 0.1,
                        }
                    ]
                }
            )
        )

    output = tmp_path / "figures"
    create_results(experiments, artifacts, output)
    assert {path.name for path in output.glob("*.png")} == {
        "comparison.png",
        "confusion.png",
        "train_loss.png",
        "val_f1.png",
        "calibration.png",
        "val_loss.png",
    }

    records: list[BenchmarkRecord] = [
        {"method": method, "gold": "allow", "prediction": "allow", "latency_ms": 1.0}
        for method in ("guard", "test_model")
    ]
    result: BenchmarkResult = {
        "metadata": {"split": "test", "n": 1},
        "summaries": [summarize(records, method) for method in ("guard", "test_model")],
        "records": records,
    }
    report = tmp_path / "comparison.json"
    report.write_text(json.dumps(result))
    create_pretrained(report, output / "pretrained")
    assert {path.name for path in (output / "pretrained").glob("*.png")} == {
        "accuracy_latency.png",
        "quality_vs_latency.png",
    }
