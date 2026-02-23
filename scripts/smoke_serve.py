#!/usr/bin/env python3
"""Smoke-check an OpenAI-compatible serving endpoint.

Checks:
1. /health returns HTTP 200
2. /v1/models returns HTTP 200 and at least one model entry
"""

from __future__ import annotations

from typing import Any

import click
import httpx

from forge.utils.logging import setup_logging


def _validate_models_payload(payload: dict[str, Any], expected_model: str | None) -> None:
    data = payload.get("data")
    if not isinstance(data, list) or len(data) == 0:
        raise click.ClickException("Smoke check failed: /v1/models returned no model entries.")

    model_ids = [
        item.get("id")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if not model_ids:
        raise click.ClickException("Smoke check failed: /v1/models payload missing model ids.")

    if expected_model and expected_model not in model_ids:
        raise click.ClickException(
            "Smoke check failed: expected model "
            f"'{expected_model}' not found. Available: {', '.join(model_ids)}"
        )


@click.command()
@click.option(
    "--base-url",
    envvar="FORGE_SERVE_BASE_URL",
    required=True,
    help="Serving base URL, e.g. http://10.34.9.233:18000",
)
@click.option(
    "--expected-model",
    envvar="FORGE_SERVE_EXPECT_MODEL",
    default=None,
    help="Optional exact model id expected in /v1/models.",
)
@click.option(
    "--api-key",
    envvar="FORGE_SERVE_API_KEY",
    default=None,
    help="Optional API key for Bearer auth (or set FORGE_SERVE_API_KEY).",
)
@click.option("--timeout", default=10, show_default=True, help="Request timeout (seconds).")
def main(base_url: str, expected_model: str | None, api_key: str | None, timeout: int) -> None:
    """Run endpoint smoke checks and exit non-zero on failure."""
    setup_logging("INFO")
    base = base_url.rstrip("/")
    health_url = f"{base}/health"
    models_url = f"{base}/v1/models"

    with httpx.Client(timeout=timeout) as client:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        health_resp = client.get(health_url, headers=headers)
        if health_resp.status_code != 200:
            raise click.ClickException(
                f"Smoke check failed: {health_url} returned {health_resp.status_code}."
            )

        models_resp = client.get(models_url, headers=headers)
        if models_resp.status_code != 200:
            raise click.ClickException(
                f"Smoke check failed: {models_url} returned {models_resp.status_code}."
            )

        payload = models_resp.json()
        if not isinstance(payload, dict):
            raise click.ClickException("Smoke check failed: /v1/models returned non-object JSON.")
        _validate_models_payload(payload, expected_model)

    click.echo(f"Serving smoke check passed for {base}")


if __name__ == "__main__":
    main()
