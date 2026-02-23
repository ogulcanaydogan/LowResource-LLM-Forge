"""vLLM inference server wrapper with health checks."""

from __future__ import annotations

import subprocess
import sys
import time

import httpx

from forge.utils.config import ServingConfig
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class VLLMServer:
    """Manage a vLLM OpenAI-compatible inference server."""

    def __init__(self, config: ServingConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None

    def start(self, wait: bool = True, timeout: int = 120) -> None:
        """Start vLLM server as a subprocess."""
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_path,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.config.max_model_len),
            "--max-num-seqs", str(self.config.max_num_seqs),
            "--dtype", self.config.dtype,
        ]

        if self.config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.config.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.config.enforce_eager:
            cmd.append("--enforce-eager")
        if self.config.api_key:
            cmd.extend(["--api-key", self.config.api_key])

        logger.info("starting_vllm", model=self.config.model_path, port=self.config.port)
        self._process = subprocess.Popen(cmd)

        if wait:
            self._wait_for_ready(timeout)

    def _wait_for_ready(self, timeout: int) -> None:
        """Wait for the server to become healthy."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.health_check():
                logger.info("vllm_ready", port=self.config.port)
                return
            time.sleep(2)
        logger.warning("vllm_startup_timeout", timeout=timeout)

    def health_check(self) -> bool:
        """Check if server is responding."""
        host = "127.0.0.1" if self.config.host in {"0.0.0.0", "::"} else self.config.host
        try:
            r = httpx.get(
                f"http://{host}:{self.config.port}/health",
                timeout=5,
            )
            return r.status_code == 200
        except httpx.RequestError:
            return False

    def stop(self) -> None:
        """Stop the server."""
        if self._process:
            logger.info("stopping_vllm")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL."""
        return f"http://{self.config.host}:{self.config.port}/v1"
