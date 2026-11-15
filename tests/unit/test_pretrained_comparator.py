from systemone_guard.benchmarking.pretrained import parse_label, summarize, user_prompt
from systemone_guard.benchmarking.types import BenchmarkRecord
from systemone_guard.types import Example


def test_strict_parser() -> None:
    assert parse_label("ALLOW") == "allow"
    assert parse_label("review. ") == "review"
    assert parse_label("BLOCK\n") == "block"
    assert parse_label("ALLOW or BLOCK") is None
    assert parse_label("I think ALLOW") is None
    assert parse_label("") is None


def test_summarize_rejects_invalid() -> None:
    x: list[BenchmarkRecord] = [
        {"method": "llm", "gold": "allow", "prediction": "allow", "latency_ms": 3},
        {"method": "llm", "gold": "block", "prediction": None, "latency_ms": 5},
    ]
    out = summarize(x, "llm")
    assert out is not None
    assert out["accuracy"] == 0.5 and out["invalid_rate"] == 0.5
    assert out["median_latency_ms"] == 4


def test_prompt_contains_authoritative_fields(sample: Example) -> None:
    text = user_prompt(sample)
    assert "AUTHORIZATION:" in text and "UNTRUSTED_CONTEXT:" in text
