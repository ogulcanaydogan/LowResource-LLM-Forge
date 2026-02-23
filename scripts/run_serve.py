#!/usr/bin/env python3
"""Run vLLM serving process from YAML config."""

from __future__ import annotations

import time

import click

from forge.serving.vllm_server import VLLMServer
from forge.utils.config import load_serving_config
from forge.utils.logging import setup_logging
from forge.utils.runtime_guard import enforce_remote_execution


@click.command()
@click.option(
    "--config",
    default="configs/serving/vllm_dgx.yaml",
    show_default=True,
    type=click.Path(exists=True),
    help="Serving config YAML.",
)
@click.option("--timeout", default=120, show_default=True, help="Health-check timeout in seconds.")
@click.option("--no-wait", is_flag=True, help="Start server without waiting for health check.")
@click.option(
    "--api-key",
    envvar="FORGE_SERVE_API_KEY",
    default=None,
    help="Optional API key for OpenAI endpoint auth (or set FORGE_SERVE_API_KEY).",
)
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def main(config: str, timeout: int, no_wait: bool, api_key: str | None, verbose: bool) -> None:
    """Start vLLM server and keep process attached."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    try:
        enforce_remote_execution("serve")
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    cfg = load_serving_config(config)
    if api_key:
        cfg.api_key = api_key
    server = VLLMServer(cfg)

    click.echo(f"Model: {cfg.model_path}")
    click.echo(f"Host: {cfg.host}")
    click.echo(f"Port: {cfg.port}")
    click.echo()

    try:
        server.start(wait=not no_wait, timeout=timeout)
        click.echo(f"vLLM ready at {server.base_url}")
        click.echo("Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nStopping server...")
    finally:
        server.stop()
        click.echo("Server stopped.")


if __name__ == "__main__":
    main()
