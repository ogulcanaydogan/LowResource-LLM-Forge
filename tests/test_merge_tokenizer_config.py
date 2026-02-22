"""Tests for tokenizer config compatibility patching."""

from __future__ import annotations

import json
from pathlib import Path

from forge.training.merge import patch_tokenizer_config_for_vllm


def _write_json(path: Path, data: dict[str, object]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _read_json(path: Path) -> dict[str, object]:
    with open(path) as f:
        return json.load(f)


def test_patch_tokenizer_config_for_vllm_rewrites_bad_class(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenizer_config.json"
    _write_json(
        config_path,
        {
            "tokenizer_class": "TokenizersBackend",
            "legacy": False,
        },
    )

    changed = patch_tokenizer_config_for_vllm(config_path)
    data = _read_json(config_path)

    assert changed is True
    assert data["tokenizer_class"] == "LlamaTokenizer"
    assert data["legacy"] is False


def test_patch_tokenizer_config_for_vllm_noop_for_compatible_class(tmp_path: Path) -> None:
    config_path = tmp_path / "tokenizer_config.json"
    _write_json(config_path, {"tokenizer_class": "LlamaTokenizer"})

    changed = patch_tokenizer_config_for_vllm(config_path)
    data = _read_json(config_path)

    assert changed is False
    assert data["tokenizer_class"] == "LlamaTokenizer"
