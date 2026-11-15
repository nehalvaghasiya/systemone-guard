"""The spacing pass must preserve Python and remain safe to run repeatedly."""

import subprocess
import sys
from pathlib import Path

from scripts.format_sections import ast_signature, format_source


def test_spacing_preserves_syntax_literals_and_attached_comments() -> None:
    source = (
        "def decide(flag):\n"
        "    first = 1\n"
        "    second = 2  # type: ignore[assignment]\n"
        "    text = '''literal\n"
        "    if flag:\n"
        "        return unchanged\n"
        "    '''\n"
        "    # Explain the guard.\n"
        "    if flag:\n"
        "        value = first + second\n"
        "        return value, text\n"
        "    return second, text\n"
    )
    formatted = format_source(source)
    assert ast_signature(formatted) == ast_signature(source)
    assert "    first = 1\n    second = 2" in formatted
    assert "\n\n    # Explain the guard.\n    if flag:" in formatted
    assert "'''literal\n    if flag:\n        return unchanged\n    '''" in formatted
    assert format_source(formatted) == formatted


def test_spacing_respects_format_suppression_and_decorators() -> None:
    source = (
        "def outer():\n"
        "    # fmt: off\n"
        "    first = 1\n"
        "    if first:\n"
        "        return first\n"
        "    # fmt: on\n"
        "    second = 2\n"
        "    @decorate\n"
        "    def inner():\n"
        "        return second\n"
        "    return inner()\n"
    )
    formatted = format_source(source)
    protected = source[source.index("    # fmt: off") : source.index("    # fmt: on")]
    assert protected in formatted
    assert "\n\n    @decorate\n    def inner():" in formatted
    assert ast_signature(formatted) == ast_signature(source)


def test_format_check_reports_changes_without_writing(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/format_sections.py"
    path = tmp_path / "example.py"
    source = "def choose(value):\n    result = value + 1\n    return result\n"
    path.write_text(source)
    result = subprocess.run(
        [sys.executable, str(script), "--check", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert path.read_text() == source
