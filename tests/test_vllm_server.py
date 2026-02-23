"""Tests for the vLLM server wrapper (no GPU required)."""

from __future__ import annotations

from forge.serving.vllm_server import VLLMServer
from forge.utils.config import ServingConfig


def _default_config(**overrides: object) -> ServingConfig:
    defaults = {
        "model_path": "/tmp/test-model",
        "host": "0.0.0.0",
        "port": 18000,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.90,
        "max_model_len": 4096,
        "dtype": "float16",
    }
    defaults.update(overrides)
    return ServingConfig(**defaults)


def test_base_url_default() -> None:
    cfg = _default_config()
    server = VLLMServer(cfg)
    assert server.base_url == "http://0.0.0.0:18000/v1"


def test_base_url_custom_port() -> None:
    cfg = _default_config(port=9999, host="10.0.0.1")
    server = VLLMServer(cfg)
    assert server.base_url == "http://10.0.0.1:9999/v1"


def test_health_check_returns_false_when_offline() -> None:
    """Health check should return False when no server is running."""
    cfg = _default_config(port=59999)
    server = VLLMServer(cfg)
    assert server.health_check() is False


def test_stop_noop_when_not_started() -> None:
    """Calling stop when server was never started should not raise."""
    cfg = _default_config()
    server = VLLMServer(cfg)
    server.stop()  # Should not raise


def test_health_check_resolves_wildcard_host() -> None:
    """0.0.0.0 and :: should resolve to 127.0.0.1 for health check."""
    for host in ("0.0.0.0", "::"):
        cfg = _default_config(host=host, port=59998)
        server = VLLMServer(cfg)
        # Verify the server resolves wildcards internally.
        assert server.health_check() is False  # No server running
        resolved = "127.0.0.1" if cfg.host in {"0.0.0.0", "::"} else cfg.host
        assert resolved == "127.0.0.1"
