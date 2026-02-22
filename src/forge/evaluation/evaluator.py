"""Evaluation orchestrator: runs all benchmarks and generates reports."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.utils.config import EvalConfig
from forge.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark."""

    name: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


class ForgeEvaluator:
    """Run evaluation benchmarks against a fine-tuned model."""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self) -> list[BenchmarkResult]:
        """Run all configured benchmarks. Returns list of results."""
        results: list[BenchmarkResult] = []

        for bench_name in self.config.benchmarks:
            logger.info("running_benchmark", benchmark=bench_name)
            try:
                result = self.run_benchmark(bench_name)
                results.append(result)
                status = "PASS" if result.passed else "FAIL"
                logger.info(
                    "benchmark_complete",
                    benchmark=bench_name,
                    score=result.score,
                    status=status,
                    duration=result.duration_seconds,
                )
            except Exception as e:
                logger.error("benchmark_failed", benchmark=bench_name, error=str(e))
                results.append(
                    BenchmarkResult(
                        name=bench_name,
                        score=0.0,
                        passed=False,
                        details={"error": str(e)},
                    )
                )

        # Save results
        from forge.evaluation.report import EvalReportGenerator

        reporter = EvalReportGenerator(results, self.output_dir)
        reporter.generate_json()
        reporter.generate_markdown()

        return results

    def run_benchmark(self, name: str) -> BenchmarkResult:
        """Run a single benchmark by name."""
        start = time.monotonic()

        if name == "mmlu_tr":
            result = self._run_mmlu_turkish()
        elif name == "perplexity":
            result = self._run_perplexity()
        elif name == "generation":
            result = self._run_generation_quality()
        else:
            raise ValueError(f"Unknown benchmark: {name}")

        result.duration_seconds = time.monotonic() - start
        return result

    def _run_mmlu_turkish(self) -> BenchmarkResult:
        """Run Turkish MMLU benchmark."""
        from forge.evaluation.benchmarks.mmlu_turkish import TurkishMMLUBenchmark

        bench = TurkishMMLUBenchmark(
            model_path=self.config.model_path,
            device=self.config.device,
        )
        details = bench.run()
        score = details.get("overall_accuracy", 0.0)
        return BenchmarkResult(
            name="mmlu_tr",
            score=score,
            passed=score >= TurkishMMLUBenchmark.PASS_THRESHOLD,
            details=details,
        )

    def _run_perplexity(self) -> BenchmarkResult:
        """Run perplexity benchmark on held-out Turkish text."""
        from forge.evaluation.benchmarks.perplexity import PerplexityBenchmark

        bench = PerplexityBenchmark(
            model_path=self.config.model_path,
            device=self.config.device,
        )
        details = bench.run()
        ppl = details.get("perplexity", float("inf"))
        # Lower perplexity is better; pass if under 50
        return BenchmarkResult(
            name="perplexity",
            score=ppl,
            passed=ppl < 50.0,
            details=details,
        )

    def _run_generation_quality(self) -> BenchmarkResult:
        """Run generation quality benchmark."""
        from forge.evaluation.benchmarks.generation_quality import (
            GenerationQualityBenchmark,
        )

        bench = GenerationQualityBenchmark(
            model_path=self.config.model_path,
            device=self.config.device,
        )
        details = bench.run()
        score = details.get("average_score", 0.0)
        return BenchmarkResult(
            name="generation",
            score=score,
            passed=score >= 3.5,
            details=details,
        )
