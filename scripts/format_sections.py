"""Add consistent blank lines between function sections without changing Python."""

from __future__ import annotations

import argparse
import ast
import io
import tokenize
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

COMPOUND = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)
EXCLUDED = {
    ".git",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "data",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def ast_signature(source: str, filename: str = "<source>") -> str:
    tree = ast.parse(source, filename=filename, type_comments=True)

    for node in ast.walk(tree):
        if isinstance(node, ast.TypeIgnore):
            node.lineno = 0

    return ast.dump(tree, include_attributes=False)


def _disabled_lines(source: str) -> set[int]:
    disabled: set[int] = set()
    start: int | None = None

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue

        marker = token.string.replace(" ", "")

        if marker == "#fmt:off":
            start = token.start[0]
        elif marker == "#fmt:on" and start is not None:
            disabled.update(range(start, token.end[0] + 1))
            start = None
        elif marker == "#fmt:skip":
            disabled.add(token.start[0])

    if start is not None:
        disabled.update(range(start, len(source.splitlines()) + 1))

    return disabled


def _needs_gap(previous: ast.stmt, current: ast.stmt) -> bool:
    if (
        isinstance(previous, ast.Expr)
        and isinstance(previous.value, ast.Constant)
        and isinstance(previous.value.value, str)
    ):
        return False

    if isinstance(previous, COMPOUND) or isinstance(current, COMPOUND):
        return True

    if isinstance(current, (ast.Return, ast.Raise)):
        return not isinstance(previous, (ast.Return, ast.Raise))

    if isinstance(current, ast.AnnAssign) and not isinstance(previous, ast.AnnAssign):
        return True

    return isinstance(previous, (ast.Import, ast.ImportFrom)) and not isinstance(
        current, (ast.Import, ast.ImportFrom)
    )


def format_source(source: str, filename: str = "<source>") -> str:
    tree = ast.parse(source, filename=filename, type_comments=True)
    lines = source.splitlines(keepends=True)
    disabled = _disabled_lines(source)

    boundaries: set[int] = set()

    def visit(node: ast.AST, in_function: bool = False) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            in_function = True
        elif isinstance(node, ast.ClassDef):
            in_function = False

        if in_function:
            for field, value in ast.iter_fields(node):
                if field not in {"body", "orelse", "finalbody"} or not isinstance(
                    value, list
                ):
                    continue

                statements = [child for child in value if isinstance(child, ast.stmt)]

                for previous, current in pairwise(statements):
                    if not _needs_gap(previous, current):
                        continue

                    previous_end = previous.end_lineno or previous.lineno
                    start = current.lineno

                    if isinstance(
                        current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                    ):
                        start = (
                            min(
                                start, *(item.lineno for item in current.decorator_list)
                            )
                            if current.decorator_list
                            else start
                        )

                    while start > previous_end + 1 and lines[
                        start - 2
                    ].lstrip().startswith("#"):
                        start -= 1

                    gap = lines[previous_end : start - 1]

                    if any(not line.strip() for line in gap):
                        continue

                    if any(
                        line in disabled
                        for line in range(
                            start, (current.end_lineno or current.lineno) + 1
                        )
                    ):
                        continue

                    boundaries.add(start - 1)

        for child in ast.iter_child_nodes(node):
            visit(child, in_function)

    visit(tree)
    newline = "\r\n" if "\r\n" in source else "\n"
    result = "".join(
        (newline if index in boundaries else "") + line
        for index, line in enumerate(lines)
    )

    if ast_signature(source, filename) != ast_signature(result, filename):
        raise ValueError(f"Spacing would change Python syntax: {filename}")

    return result


def source_files(paths: Sequence[Path]) -> list[Path]:
    files: set[Path] = set()

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

        candidates = path.rglob("*") if path.is_dir() else [path]
        files.update(
            candidate
            for candidate in candidates
            if candidate.is_file()
            and not candidate.is_symlink()
            and candidate.suffix in {".py", ".pyi"}
            and not EXCLUDED.intersection(candidate.parts)
        )

    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("src"),
            Path("scripts"),
            Path("spaces"),
            Path("tests"),
        ],
    )
    args = parser.parse_args()

    changes: list[tuple[Path, bytes]] = []

    for path in source_files(args.paths):
        original = path.read_bytes()
        encoding, _ = tokenize.detect_encoding(io.BytesIO(original).readline)
        updated = format_source(original.decode(encoding), str(path)).encode(encoding)

        if original != updated:
            changes.append((path, updated))

    for path, updated in changes:
        if args.check:
            print(f"Logical section spacing required: {path}")
        else:
            path.write_bytes(updated)

    if not args.check:
        print(f"Logical section spacing: {len(changes)} files updated")

    return int(args.check and bool(changes))


if __name__ == "__main__":
    raise SystemExit(main())
