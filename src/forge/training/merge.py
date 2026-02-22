"""Merge LoRA adapters into base model for inference deployment.

Adapted from autonomous-dev-agent/scripts/merge_lora_weights.py.
Uses the proven PeftModel.from_pretrained() + merge_and_unload() pattern.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

logger = get_logger(__name__)

_BAD_TOKENIZER_CLASS = "TokenizersBackend"
_SERVING_TOKENIZER_CLASS = "LlamaTokenizer"


def patch_tokenizer_config_for_vllm(config_path: Path) -> bool:
    """Patch incompatible tokenizer metadata for vLLM/transformers loading.

    Some tokenizers saved through remote-code backends set
    `tokenizer_class=TokenizersBackend`, which cannot be imported by
    transformers AutoTokenizer in standard serving environments.
    """
    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = json.load(f)

    if config.get("tokenizer_class") != _BAD_TOKENIZER_CLASS:
        return False

    config["tokenizer_class"] = _SERVING_TOKENIZER_CLASS
    config.setdefault("legacy", True)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return True


class LoRAMerger:
    """Merge LoRA adapters into base model."""

    def __init__(
        self,
        base_model_name: str,
        adapter_path: str | Path,
        output_path: str | Path,
    ) -> None:
        self.base_model_name = base_model_name
        self.adapter_path = Path(adapter_path)
        self.output_path = Path(output_path)

    def merge(
        self, push_to_hub: bool = False, hub_repo: str | None = None
    ) -> Path:
        """Merge LoRA adapters into base model. Returns output path."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        logger.info("loading_base_model", model=self.base_model_name)
        base_model: Any = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=self._torch_dtype(),
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("loading_lora_adapters", path=str(self.adapter_path))
        peft_model: Any = PeftModel.from_pretrained(base_model, str(self.adapter_path))

        logger.info("merging_weights")
        merged_model: Any = peft_model.merge_and_unload()

        logger.info("saving_merged_model", path=str(self.output_path))
        self.output_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(self.output_path))
        tokenizer = self._load_and_save_serving_tokenizer()

        self._save_metadata()

        if push_to_hub and hub_repo:
            logger.info("pushing_to_hub", repo=hub_repo)
            merged_model.push_to_hub(hub_repo, private=True)
            tokenizer.push_to_hub(hub_repo, private=True)

        self._cleanup()
        logger.info("merge_complete", output=str(self.output_path))
        return self.output_path

    def _load_and_save_serving_tokenizer(self) -> Any:
        """Save tokenizer artifacts in a serving-compatible form."""
        from transformers import AutoTokenizer

        attempts = [
            {"trust_remote_code": False, "use_fast": True},
            {"trust_remote_code": False},
            {"trust_remote_code": True},
        ]
        last_error: Exception | None = None

        for kwargs in attempts:
            try:
                logger.info("loading_tokenizer", model=self.base_model_name, **kwargs)
                tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model_name,
                    **kwargs,
                )
                tokenizer.save_pretrained(str(self.output_path))
                config_path = self.output_path / "tokenizer_config.json"
                if patch_tokenizer_config_for_vllm(config_path):
                    logger.warning(
                        "patched_tokenizer_config",
                        path=str(config_path),
                        from_class=_BAD_TOKENIZER_CLASS,
                        to_class=_SERVING_TOKENIZER_CLASS,
                    )
                return tokenizer
            except Exception as exc:  # pragma: no cover - depends on remote model/tokenizer
                last_error = exc
                logger.warning("tokenizer_load_attempt_failed", error=str(exc), **kwargs)

        raise RuntimeError("Failed to load tokenizer for merged model output.") from last_error

    def _save_metadata(self) -> None:
        """Save merge provenance metadata."""
        metadata = {
            "base_model": self.base_model_name,
            "adapter_path": str(self.adapter_path),
            "merged": True,
            "merge_method": "peft_merge_and_unload",
        }
        with open(self.output_path / "merge_info.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _cleanup(self) -> None:
        """Free GPU memory."""
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _torch_dtype() -> Any:
        import torch

        return torch.float16
