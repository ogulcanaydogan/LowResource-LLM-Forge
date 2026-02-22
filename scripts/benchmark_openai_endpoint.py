#!/usr/bin/env python3
"""Run a lightweight benchmark pass against an OpenAI-compatible endpoint."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import click
import httpx

DEFAULT_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "chat-1",
        "category": "chat",
        "prompt": "Explain what overfitting is in plain Turkish with one example.",
        "max_tokens": 140,
        "temperature": 0.2,
    },
    {
        "id": "chat-2",
        "category": "chat",
        "prompt": "Write a concise Turkish customer support reply for a delayed shipment.",
        "max_tokens": 120,
        "temperature": 0.3,
    },
    {
        "id": "coding-1",
        "category": "coding",
        "prompt": "Write a Python function that returns Fibonacci numbers up to n.",
        "max_tokens": 220,
        "temperature": 0.1,
    },
    {
        "id": "coding-2",
        "category": "coding",
        "prompt": (
            "Refactor this idea into Python pseudocode: retry HTTP request "
            "with exponential backoff."
        ),
        "max_tokens": 220,
        "temperature": 0.1,
    },
    {
        "id": "reasoning-1",
        "category": "reasoning",
        "prompt": "If all A are B and some B are C, can we conclude some A are C? Explain briefly.",
        "max_tokens": 140,
        "temperature": 0.1,
    },
    {
        "id": "reasoning-2",
        "category": "reasoning",
        "prompt": "A train leaves at 08:15 and arrives at 11:45. What is the travel duration?",
        "max_tokens": 80,
        "temperature": 0.0,
    },
]


def _load_prompts(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEFAULT_PROMPTS

    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise click.ClickException("Prompts file must contain a JSON list.")
    prompts: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise click.ClickException(f"Prompt entry at index {idx} is not an object.")
        if "prompt" not in item:
            raise click.ClickException(f"Prompt entry at index {idx} is missing 'prompt'.")
        prompts.append(
            {
                "id": item.get("id", f"prompt-{idx + 1}"),
                "category": item.get("category", "uncategorized"),
                "prompt": item["prompt"],
                "max_tokens": int(item.get("max_tokens", 160)),
                "temperature": float(item.get("temperature", 0.2)),
            }
        )
    return prompts


def _resolve_model(client: httpx.Client, base_url: str, model: str | None) -> str:
    if model:
        return model

    resp = client.get(f"{base_url}/v1/models")
    resp.raise_for_status()
    data = resp.json()
    entries = data.get("data", [])
    if not entries:
        raise click.ClickException("No models returned by /v1/models.")
    model_id = entries[0].get("id")
    if not isinstance(model_id, str) or not model_id:
        raise click.ClickException("Could not resolve model id from /v1/models.")
    return model_id


@click.command()
@click.option("--base-url", required=True, help="Endpoint base URL, e.g. http://127.0.0.1:18000")
@click.option("--model", default=None, help="Model id. If omitted, auto-detected from /v1/models.")
@click.option(
    "--prompts-file",
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    default=None,
    help="Optional JSON prompt list.",
)
@click.option("--output", required=True, type=click.Path(path_type=str), help="Output JSON path.")
@click.option("--runs-per-prompt", default=1, show_default=True, type=int)
@click.option("--timeout-seconds", default=90.0, show_default=True, type=float)
def main(
    base_url: str,
    model: str | None,
    prompts_file: str | None,
    output: str,
    runs_per_prompt: int,
    timeout_seconds: float,
) -> None:
    """Benchmark an OpenAI-compatible endpoint with a fixed prompt set."""
    base = base_url.rstrip("/")
    prompts = _load_prompts(prompts_file)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(UTC).isoformat()

    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=timeout_seconds) as client:
        health = client.get(f"{base}/health")
        if health.status_code != 200:
            raise click.ClickException(
                f"Health check failed: {base}/health => {health.status_code}"
            )

        model_id = _resolve_model(client, base, model)
        click.echo(f"Benchmarking {base} with model={model_id}")

        for prompt_cfg in prompts:
            for run_idx in range(runs_per_prompt):
                payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt_cfg["prompt"]}],
                    "temperature": prompt_cfg["temperature"],
                    "max_tokens": prompt_cfg["max_tokens"],
                }
                t0 = time.perf_counter()
                status_code = 0
                completion_text = ""
                error = ""
                try:
                    resp = client.post(f"{base}/v1/chat/completions", json=payload)
                    status_code = resp.status_code
                    if status_code == 200:
                        body = resp.json()
                        choice = (body.get("choices") or [{}])[0]
                        message = choice.get("message", {})
                        completion_text = str(message.get("content", ""))[:2000]
                    else:
                        error = resp.text[:500]
                except Exception as exc:  # noqa: BLE001
                    error = str(exc)
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                ok = status_code == 200
                results.append(
                    {
                        "id": prompt_cfg["id"],
                        "category": prompt_cfg["category"],
                        "run_index": run_idx + 1,
                        "status_code": status_code,
                        "ok": ok,
                        "latency_ms": latency_ms,
                        "completion_chars": len(completion_text),
                        "error": error,
                    }
                )
                click.echo(
                    f"  {prompt_cfg['id']} run={run_idx + 1} ok={ok} "
                    f"status={status_code} latency_ms={latency_ms}"
                )

    categories = sorted({item["category"] for item in results})
    by_category: dict[str, dict[str, Any]] = {}
    for category in categories:
        rows = [row for row in results if row["category"] == category]
        latencies = [row["latency_ms"] for row in rows if row["ok"]]
        by_category[category] = {
            "count": len(rows),
            "success_count": sum(1 for row in rows if row["ok"]),
            "avg_latency_ms": round(mean(latencies), 2) if latencies else None,
        }

    all_latencies = [row["latency_ms"] for row in results if row["ok"]]
    summary = {
        "count": len(results),
        "success_count": sum(1 for row in results if row["ok"]),
        "avg_latency_ms": round(mean(all_latencies), 2) if all_latencies else None,
        "by_category": by_category,
    }

    report = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "base_url": base,
        "model": model_id,
        "runs_per_prompt": runs_per_prompt,
        "prompt_count": len(prompts),
        "prompts": prompts,
        "results": results,
        "summary": summary,
    }
    output_path.write_text(json.dumps(report, indent=2))
    click.echo(f"Saved benchmark: {output_path}")


if __name__ == "__main__":
    main()
