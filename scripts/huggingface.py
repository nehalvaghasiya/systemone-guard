"""Prepare an offline Hugging Face Space or explicitly publish model files."""

import argparse
import os
import shutil
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib as toml
else:
    import tomli as toml


def prepare(root: Path, output: Path, checkpoint: Path | None = None) -> Path:
    checkpoint = checkpoint or root / "artifacts/guard_best/model.pt"

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint required for Space export: {checkpoint}")

    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        root / "src/systemone_guard",
        output / "systemone_guard",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(root / "spaces/app.py", output / "app.py")
    weights = output / "artifacts/guard_best/model.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, weights)
    project = toml.loads((root / "pyproject.toml").read_text())["project"]
    requirements = project["dependencies"] + project["optional-dependencies"]["demo"]
    (output / "requirements.txt").write_text("\n".join(requirements) + "\n")
    (output / "README.md").write_text(SPACE_README)

    return output


SPACE_README = "---\ntitle: SystemOne Guard\nemoji: 🛡️\ncolorFrom: blue\ncolorTo: gray\nsdk: gradio\napp_file: app.py\npinned: false\n---\n\n# SystemOne Guard\n\nSingle-pass typed probabilistic decision network trained from scratch on synthetic examples. Research only, **not a reliable security detector**. Novel manually written adversarial cases were almost all missed. See the main repository documentation for experiment provenance, tests, limitations, and benchmarks.\n"


def publish(root: Path, repo: str, checkpoint: Path, public: bool = False) -> None:
    token = os.environ.get("HF_TOKEN")

    if not token:
        raise SystemExit("Set HF_TOKEN before uploading. No token is written to disk.")

    from huggingface_hub import HfApi

    source = root / "src/systemone_guard"
    api = HfApi(token=token)
    api.create_repo(repo, repo_type="model", private=not public, exist_ok=True)
    api.upload_file(
        repo_id=repo, path_or_fileobj=str(checkpoint), path_in_repo="model.pt"
    )
    api.upload_file(
        repo_id=repo,
        path_or_fileobj=str(root / "hf_model/README.md"),
        path_in_repo="README.md",
    )

    for path in sorted(source.rglob("*")):
        if path.is_file() and (path.suffix == ".py" or path.name == "py.typed"):
            api.upload_file(
                repo_id=repo,
                path_or_fileobj=str(path),
                path_in_repo="systemone_guard/" + path.relative_to(source).as_posix(),
            )

    print("Uploaded checkpoint, source, and model card to", repo)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    space = commands.add_parser(
        "prepare", help="Prepare a local Space export without uploading"
    )
    space.add_argument("--out", type=Path, default=root / "dist/systemone-guard-space")
    space.add_argument("--checkpoint", type=Path)
    upload = commands.add_parser(
        "publish", help="Upload using the publish extra and HF_TOKEN"
    )
    upload.add_argument("--repo", required=True)
    upload.add_argument("--public", action="store_true")
    upload.add_argument(
        "--checkpoint", type=Path, default=root / "artifacts/guard_best/model.pt"
    )
    args = parser.parse_args()

    if args.command == "prepare":
        print(prepare(root, args.out, args.checkpoint))
    else:
        publish(root, args.repo, args.checkpoint, args.public)


if __name__ == "__main__":
    main()
