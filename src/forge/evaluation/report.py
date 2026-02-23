"""Markdown and JSON report generation for eval results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from forge.utils.logging import get_logger

if TYPE_CHECKING:
    from forge.evaluation.evaluator import BenchmarkResult

logger = get_logger(__name__)


class EvalReportGenerator:
    """Writes results.json and report.md from benchmark runs."""

    def __init__(self, results: list[BenchmarkResult], output_dir: Path) -> None:
        self.results = results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_json(self) -> Path:
        """Generate machine-readable JSON results."""
        path = self.output_dir / "results.json"
        data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "benchmarks": [
                {
                    "name": r.name,
                    "score": r.score,
                    "passed": r.passed,
                    "duration_seconds": r.duration_seconds,
                    "details": r.details,
                }
                for r in self.results
            ],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)  # default=str handles datetime
        logger.info("json_report_written", path=str(path))
        return path

    def generate_markdown(self) -> Path:
        """Generate a Markdown evaluation report."""
        path = self.output_dir / "report.md"
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        lines = [
            "# Evaluation Report",
            "",
            f"**Date:** {timestamp}",
            f"**Result:** {passed}/{total} benchmarks passed",
            "",
            "## Benchmark Results",
            "",
            "| Benchmark | Score | Status | Duration |",
            "|-----------|-------|--------|----------|",
        ]

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            duration = f"{r.duration_seconds:.1f}s"
            lines.append(f"| {r.name} | {r.score:.4f} | {status} | {duration} |")

        lines.extend(["", "## Details", ""])

        for r in self.results:
            lines.append(f"### {r.name}")
            lines.append("")
            for key, value in r.details.items():
                if key == "samples":
                    continue  # too verbose for the summary
                lines.append(f"- **{key}:** {value}")
            lines.append("")

        content = "\n".join(lines) + "\n"
        with open(path, "w") as f:
            f.write(content)
        logger.info("markdown_report_written", path=str(path))
        return path
