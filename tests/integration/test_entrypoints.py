"""CLI compatibility, import safety, and generated deliverables."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from scripts.huggingface import prepare
from scripts.plots import create_plots

from systemone_guard.benchmarking.analysis import METHODS, compare_records
from systemone_guard.benchmarking.comparison import summarize
from systemone_guard.benchmarking.reporting import load_reports
from systemone_guard.benchmarking.types import BenchmarkRecord, BenchmarkResult
from systemone_guard.inference import DecisionEngine
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
    [
        "data",
        "train",
        "robust_train",
        "evaluate",
        "inference",
        "benchmarking.comparison",
        "benchmarking.finetune",
        "benchmarking.reporting",
    ],
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
    assert (output / "systemone_guard/benchmarking/comparison.py").is_file()
    assert (output / "systemone_guard/py.typed").is_file()
    assert (
        output / "artifacts/guard_scaled/model.pt"
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


def test_reports_and_all_six_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib-cache"))
    experiments = tmp_path / "experiments"
    experiments.mkdir()

    records: list[BenchmarkRecord] = [
        {
            "id": "example-1",
            "method": method,
            "gold": "allow",
            "prediction": "allow",
            "latency_ms": latency,
            "probabilities": {"allow": 0.8, "review": 0.1, "block": 0.1},
            "auxiliary_gold": {"sensitive": 0, "approval_needed": 0, "attack": 0},
            "auxiliary_probabilities": {
                "sensitive": 0.1,
                "approval_needed": 0.1,
                "attack": 0.1,
            },
        }
        for method, latency in zip(METHODS, (1.0, 10.0), strict=True)
    ]
    result: BenchmarkResult = {
        "metadata": {"split": "test", "n": 1, "seed": 42},
        "summaries": [summarize(records, method) for method in METHODS],
        "records": records,
        "comparison": compare_records(records),
    }

    for split in ("test", "challenge", "manual_probes", "generalization"):
        (experiments / f"finetuned_{split}.json").write_text(json.dumps(result))

    (experiments / "guard_training.json").write_text(
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
    (experiments / "smollm2_finetuning.json").write_text(
        json.dumps(
            {
                "history": [
                    {
                        "epoch": 1,
                        "training_loss": 0.2,
                        "validation": {"macro_f1": 1.0, "nll": 0.1},
                    }
                ]
            }
        )
    )
    output = tmp_path / "figures"
    create_plots(experiments, output)
    assert {path.name for path in output.glob("*.png")} == {
        "comparison.png",
        "latency.png",
        "calibration.png",
        "confusion.png",
        "auxiliary_accuracy.png",
        "training.png",
    }
    assert all(path.stat().st_size > 10000 for path in output.glob("*.png"))
    result["comparison"]["accuracy_gain_pp"] = 99.0
    (experiments / "finetuned_test.json").write_text(json.dumps(result))

    with pytest.raises(ValueError, match="differ from the raw records"):
        load_reports(experiments)
