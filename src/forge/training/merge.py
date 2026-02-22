"""Merge LoRA adapters into base model for inference deployment.

Adapted from autonomous-dev-agent/scripts/merge_lora_weights.py.
Uses the proven PeftModel.from_pretrained() + merge_and_unload() pattern.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from forge.utils.logging import get_logger

logger = get_logger(__name__)


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
        logger.info("loading_tokenizer", model=self.base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        logger.info("loading_base_model", model=self.base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("loading_lora_adapters", path=str(self.adapter_path))
        model = PeftModel.from_pretrained(model, str(self.adapter_path))

        logger.info("merging_weights")
        model = model.merge_and_unload()

        logger.info("saving_merged_model", path=str(self.output_path))
        self.output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self.output_path))
        tokenizer.save_pretrained(str(self.output_path))

        self._save_metadata()

        if push_to_hub and hub_repo:
            logger.info("pushing_to_hub", repo=hub_repo)
            model.push_to_hub(hub_repo, private=True)
            tokenizer.push_to_hub(hub_repo, private=True)

        self._cleanup()
        logger.info("merge_complete", output=str(self.output_path))
        return self.output_path

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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
