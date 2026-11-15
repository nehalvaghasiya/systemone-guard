"""Explicit wheel and optional-adapter checks outside the default test suite."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from systemone_guard.inference import DecisionEngine
from systemone_guard.types import Example, decision_input

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.packaging
def test_isolated_wheel(
    checkpoint: Path,
    sample: Example,
    tmp_path: Path,
) -> None:
    wheels = list((ROOT / "dist").glob("systemone_guard-*.whl"))
    assert wheels, "Build a wheel first: make build"
    wheel = max(wheels, key=lambda path: path.stat().st_mtime)
    environment = tmp_path / "venv"
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)}
    env.pop("VIRTUAL_ENV", None)
    uv = os.environ.get("JEV_UV", "uv")
    subprocess.run(
        [
            uv,
            "sync",
            "--locked",
            "--no-install-project",
            "--extra",
            "cpu",
            "--python",
            sys.executable,
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(
        [uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        cwd=tmp_path,
        env=env,
        check=True,
    )
    code = """
from pathlib import Path
import sys
import systemone_guard
from systemone_guard.benchmarking.pretrained import parse_label
package = Path(systemone_guard.__file__).resolve()
assert Path(sys.prefix) in package.parents, package
assert (package.parent / 'py.typed').is_file()
assert parse_label('ALLOW') == 'allow'
"""
    subprocess.run([str(python), "-c", code], cwd=tmp_path, env=env, check=True)
    inputs = decision_input(sample)
    arguments = [
        argument
        for name, value in inputs.items()
        for argument in (f"--{name}", str(value))
    ]
    result = subprocess.run(
        [
            str(python),
            "-m",
            "systemone_guard.inference",
            "--checkpoint",
            str(checkpoint),
            *arguments,
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == DecisionEngine(checkpoint).decide(**inputs)


@pytest.mark.optional
def test_optional_adapters(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    for name in (
        "gradio",
        "transformers",
        "huggingface_hub",
        "accelerate",
        "sentencepiece",
    ):
        importlib.import_module(name)

    from huggingface_hub import HfApi
    from spaces.app import create_demo
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert callable(AutoModelForCausalLM.from_pretrained)
    assert callable(AutoTokenizer.from_pretrained)
    assert callable(HfApi.upload_file)
    demo = create_demo(checkpoint)

    try:
        assert demo.config["components"]
    finally:
        demo.close()
