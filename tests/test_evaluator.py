"""Tests for the evaluation framework."""

from __future__ import annotations

from pathlib import Path

from forge.evaluation.evaluator import BenchmarkResult
from forge.evaluation.report import EvalReportGenerator
from forge.utils.config import EvalConfig


def test_benchmark_result_creation() -> None:
    result = BenchmarkResult(
        name="test_bench",
        score=0.85,
        passed=True,
        details={"accuracy": 0.85},
        duration_seconds=10.5,
    )
    assert result.name == "test_bench"
    assert result.score == 0.85
    assert result.passed is True


def test_eval_config_defaults() -> None:
    cfg = EvalConfig(model_path="test/model")
    assert len(cfg.benchmarks) == 3
    assert "mmlu_tr" in cfg.benchmarks
    assert "perplexity" in cfg.benchmarks
    assert "generation" in cfg.benchmarks


def test_eval_config_custom_benchmarks() -> None:
    cfg = EvalConfig(model_path="test/model", benchmarks=["perplexity"])
    assert cfg.benchmarks == ["perplexity"]


def test_report_generator_json(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(name="bench1", score=0.8, passed=True, duration_seconds=5.0),
        BenchmarkResult(name="bench2", score=0.3, passed=False, duration_seconds=10.0),
    ]
    reporter = EvalReportGenerator(results, tmp_path)
    path = reporter.generate_json()

    assert path.exists()
    import json
    with open(path) as f:
        data = json.load(f)
    assert data["summary"]["total"] == 2
    assert data["summary"]["passed"] == 1
    assert data["summary"]["failed"] == 1


def test_report_generator_markdown(tmp_path: Path) -> None:
    results = [
        BenchmarkResult(
            name="bench1", score=0.8, passed=True,
            details={"note": "good"}, duration_seconds=5.0,
        ),
    ]
    reporter = EvalReportGenerator(results, tmp_path)
    path = reporter.generate_markdown()

    assert path.exists()
    content = path.read_text()
    assert "# Evaluation Report" in content
    assert "bench1" in content
    assert "PASS" in content
