#!/usr/bin/env python3
"""Compare benchmark outputs and generate markdown/json summaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click


@dataclass
class Metrics:
    label: str
    source: str
    timestamp_utc: str
    model: str
    base_url: str
    total: int
    success: int
    success_rate_pct: float
    avg_latency_ms: float | None
    p95_latency_ms: float | None


@dataclass
class CategoryMetrics:
    category: str
    total: int
    success: int
    avg_latency_ms: float | None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _fmt_number(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_delta(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}{suffix}"


def _load_report(path: Path, label: str) -> tuple[Metrics, dict[str, CategoryMetrics]]:
    payload = json.loads(path.read_text())

    results = payload.get("results", [])
    if not isinstance(results, list):
        raise click.ClickException(f"Invalid report format: results is not a list ({path})")

    total = len(results)
    success = sum(1 for row in results if bool(row.get("ok")))
    success_rate = (success / total * 100.0) if total else 0.0

    latencies = [float(row["latency_ms"]) for row in results if bool(row.get("ok"))]

    summary = payload.get("summary", {})
    avg_latency = summary.get("avg_latency_ms")
    if avg_latency is not None:
        avg_latency = float(avg_latency)
    elif latencies:
        avg_latency = sum(latencies) / len(latencies)

    p95_latency = _percentile(latencies, 0.95)

    timestamp = str(payload.get("finished_at_utc") or payload.get("started_at_utc") or "")
    if not timestamp:
        timestamp = datetime.now(UTC).isoformat()

    metrics = Metrics(
        label=label,
        source=str(path),
        timestamp_utc=timestamp,
        model=str(payload.get("model", "")),
        base_url=str(payload.get("base_url", "")),
        total=total,
        success=success,
        success_rate_pct=success_rate,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
    )

    category_map: dict[str, CategoryMetrics] = {}
    category_rows: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        category = str(row.get("category", "uncategorized"))
        category_rows.setdefault(category, []).append(row)

    for category, rows in sorted(category_rows.items()):
        c_total = len(rows)
        c_success = sum(1 for row in rows if bool(row.get("ok")))
        c_latencies = [float(row["latency_ms"]) for row in rows if bool(row.get("ok"))]
        c_avg = (sum(c_latencies) / len(c_latencies)) if c_latencies else None
        category_map[category] = CategoryMetrics(
            category=category,
            total=c_total,
            success=c_success,
            avg_latency_ms=c_avg,
        )

    return metrics, category_map


def _build_markdown(
    current: Metrics,
    current_categories: dict[str, CategoryMetrics],
    baseline: Metrics | None,
    baseline_categories: dict[str, CategoryMetrics] | None,
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = []
    lines.append("# Endpoint Benchmark Compare")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Current source: `{current.source}`")
    if baseline:
        lines.append(f"Baseline source: `{baseline.source}`")
    else:
        lines.append("Baseline source: _none (first comparable run)_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Delta |")
    lines.append("|---|---:|---:|---:|")

    baseline_avg = baseline.avg_latency_ms if baseline else None
    baseline_p95 = baseline.p95_latency_ms if baseline else None
    baseline_sr = baseline.success_rate_pct if baseline else None

    delta_avg = (
        current.avg_latency_ms - baseline_avg
        if current.avg_latency_ms is not None and baseline_avg is not None
        else None
    )
    delta_p95 = (
        current.p95_latency_ms - baseline_p95
        if current.p95_latency_ms is not None and baseline_p95 is not None
        else None
    )
    delta_sr = (
        current.success_rate_pct - baseline_sr
        if baseline_sr is not None
        else None
    )

    lines.append(
        "| Avg latency (ms) | "
        f"{_fmt_number(baseline_avg)} | {_fmt_number(current.avg_latency_ms)} | "
        f"{_fmt_delta(delta_avg)} |"
    )
    lines.append(
        "| P95 latency (ms) | "
        f"{_fmt_number(baseline_p95)} | {_fmt_number(current.p95_latency_ms)} | "
        f"{_fmt_delta(delta_p95)} |"
    )
    lines.append(
        "| Success rate (%) | "
        f"{_fmt_number(baseline_sr)} | {_fmt_number(current.success_rate_pct)} | "
        f"{_fmt_delta(delta_sr)} |"
    )
    lines.append(
        "| Successful requests | "
        f"{baseline.success if baseline else 'n/a'} / {baseline.total if baseline else 'n/a'} | "
        f"{current.success} / {current.total} | "
        f"{(current.success - baseline.success) if baseline else 'n/a'} |"
    )
    lines.append("")

    lines.append("## Trend")
    lines.append("")
    lines.append(
        "| Timestamp (UTC) | Label | Avg latency (ms) | "
        "P95 latency (ms) | Success rate (%) | Success/Total |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    if baseline:
        lines.append(
            f"| {baseline.timestamp_utc} | {baseline.label} | "
            f"{_fmt_number(baseline.avg_latency_ms)} | {_fmt_number(baseline.p95_latency_ms)} | "
            f"{_fmt_number(baseline.success_rate_pct)} | {baseline.success}/{baseline.total} |"
        )
    lines.append(
        f"| {current.timestamp_utc} | {current.label} | "
        f"{_fmt_number(current.avg_latency_ms)} | {_fmt_number(current.p95_latency_ms)} | "
        f"{_fmt_number(current.success_rate_pct)} | {current.success}/{current.total} |"
    )
    lines.append("")

    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Baseline avg (ms) | Current avg (ms) | Delta (ms) |")
    lines.append("|---|---:|---:|---:|")

    category_names = sorted(
        set(current_categories.keys())
        | (set(baseline_categories.keys()) if baseline_categories else set())
    )
    for category in category_names:
        current_cat = current_categories.get(category)
        baseline_cat = baseline_categories.get(category) if baseline_categories else None
        current_avg = current_cat.avg_latency_ms if current_cat else None
        baseline_avg_cat = baseline_cat.avg_latency_ms if baseline_cat else None
        delta_cat = (
            current_avg - baseline_avg_cat
            if current_avg is not None and baseline_avg_cat is not None
            else None
        )
        lines.append(
            f"| {category} | {_fmt_number(baseline_avg_cat)} | "
            f"{_fmt_number(current_avg)} | {_fmt_delta(delta_cat)} |"
        )

    lines.append("")
    lines.append("## Endpoint")
    lines.append("")
    lines.append(f"- Base URL: `{current.base_url}`")
    lines.append(f"- Model: `{current.model}`")

    return "\n".join(lines) + "\n"


@click.command()
@click.option(
    "--current",
    "current_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--baseline",
    "baseline_path",
    required=False,
    type=click.Path(exists=True, path_type=Path),
    default=None,
)
@click.option("--current-label", default="current", show_default=True)
@click.option("--baseline-label", default="baseline", show_default=True)
@click.option("--output-md", required=True, type=click.Path(path_type=Path))
@click.option("--output-json", required=True, type=click.Path(path_type=Path))
def main(
    current_path: Path,
    baseline_path: Path | None,
    current_label: str,
    baseline_label: str,
    output_md: Path,
    output_json: Path,
) -> None:
    """Generate benchmark comparison markdown/json files."""
    current, current_categories = _load_report(current_path, current_label)

    baseline: Metrics | None = None
    baseline_categories: dict[str, CategoryMetrics] | None = None
    if baseline_path is not None:
        baseline, baseline_categories = _load_report(baseline_path, baseline_label)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    markdown = _build_markdown(current, current_categories, baseline, baseline_categories)
    output_md.write_text(markdown)

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "current": current.__dict__,
        "baseline": baseline.__dict__ if baseline else None,
        "current_categories": {k: v.__dict__ for k, v in current_categories.items()},
        "baseline_categories": (
            {k: v.__dict__ for k, v in baseline_categories.items()}
            if baseline_categories
            else None
        ),
    }
    if baseline:
        payload["delta"] = {
            "avg_latency_ms": (
                current.avg_latency_ms - baseline.avg_latency_ms
                if current.avg_latency_ms is not None and baseline.avg_latency_ms is not None
                else None
            ),
            "p95_latency_ms": (
                current.p95_latency_ms - baseline.p95_latency_ms
                if current.p95_latency_ms is not None and baseline.p95_latency_ms is not None
                else None
            ),
            "success_rate_pct": current.success_rate_pct - baseline.success_rate_pct,
            "successful_requests": current.success - baseline.success,
        }

    output_json.write_text(json.dumps(payload, indent=2))

    click.echo(f"Wrote compare markdown: {output_md}")
    click.echo(f"Wrote compare json: {output_json}")


if __name__ == "__main__":
    main()
