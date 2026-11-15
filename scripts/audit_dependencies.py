"""Audit installed dependencies, including PyTorch's CPU wheel versions."""

import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    requirements = []

    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        normalized = name.lower().replace("_", "-")

        if normalized == "systemone-guard":
            continue

        version = distribution.version

        if normalized == "torch":
            # CPU wheels append +cpu to the same upstream release. Advisory
            # databases identify it by the public version published on PyPI.
            version = version.split("+", 1)[0]

        requirements.append(f"{name}=={version}")

    with tempfile.TemporaryDirectory(prefix="systemone-audit-") as folder:
        directory = Path(folder)
        path = directory / "requirements.txt"
        path.write_text("\n".join(sorted(requirements)) + "\n")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--no-deps",
                "--disable-pip",
                "--cache-dir",
                str(directory / "cache"),
                "-r",
                str(path),
                *sys.argv[1:],
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
