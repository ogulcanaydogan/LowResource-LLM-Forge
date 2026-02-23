"""Unsloth QLoRA training orchestrator for low-resource language fine-tuning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from datasets import load_dataset

from forge.utils.config import TrainingConfig
from forge.utils.logging import get_logger

logger = get_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in _TRUE_VALUES)


class ForgeTrainer:
    """Orchestrate QLoRA fine-tuning with Unsloth on V100.

    Falls back to standard PEFT if Unsloth is not available.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.model: Any = None
        self.tokenizer: Any = None
        self._use_unsloth = True

    def setup(self) -> None:
        """Load base model with 4-bit quantization and apply LoRA adapters."""
        try:
            self._setup_unsloth()
        except ImportError:
            logger.warning("unsloth_not_available", msg="Falling back to standard PEFT")
            self._use_unsloth = False
            self._setup_peft()

    def _setup_unsloth(self) -> None:
        """Load model via Unsloth for optimized training."""
        from unsloth import FastLanguageModel

        logger.info(
            "loading_model_unsloth",
            model=self.config.model.name,
            max_seq_length=self.config.model.max_seq_length,
        )

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model.name,
            max_seq_length=self.config.model.max_seq_length,
            dtype=None,  # Auto-detect (fp16 for V100)
            load_in_4bit=self.config.quantization.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
            use_gradient_checkpointing="unsloth",
            random_state=self.config.training.seed,
        )

        logger.info("model_loaded_unsloth", trainable_params=self._count_trainable_params())

    def _setup_peft(self) -> None:
        """Load model via standard PEFT (fallback when Unsloth unavailable)."""
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info(
            "loading_model_peft",
            model=self.config.model.name,
            max_seq_length=self.config.model.max_seq_length,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = prepare_model_for_kbit_training(self.model)

        lora_bias = self.config.lora.bias.lower()
        valid_lora_bias = {"none", "all", "lora_only"}
        if lora_bias not in valid_lora_bias:
            raise ValueError(
                f"Invalid LoRA bias '{self.config.lora.bias}'. "
                "Expected one of: none, all, lora_only."
            )

        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias=cast(Literal["none", "all", "lora_only"], lora_bias),
            task_type=self.config.lora.task_type,
        )
        self.model = get_peft_model(self.model, lora_config)

        logger.info("model_loaded_peft", trainable_params=self._count_trainable_params())

    def _count_trainable_params(self) -> int:
        """Count number of trainable parameters."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _load_datasets(self) -> tuple[Any, Any]:
        """Load train and eval datasets from JSONL files."""
        train_dataset = load_dataset(
            "json", data_files=self.config.train_data_path, split="train"
        )
        eval_dataset = load_dataset(
            "json", data_files=self.config.eval_data_path, split="train"
        )
        logger.info(
            "datasets_loaded",
            train_size=len(train_dataset),
            eval_size=len(eval_dataset),
        )
        return train_dataset, eval_dataset

    def _format_prompt(self, example: dict[str, str]) -> str:
        """Format a single example into the model's chat template.

        Uses alpaca format: instruction + optional input -> output.
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            prompt = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )

        return prompt

    def train(self, resume_from_checkpoint: str | None = None) -> Path:
        """Run training. Returns path to saved adapter."""
        from trl import SFTConfig, SFTTrainer

        if self.model is None:
            raise RuntimeError("Call setup() before train()")

        train_dataset, eval_dataset = self._load_datasets()

        run_name = self.config.wandb.run_name or "forge-run"
        output_dir = Path(self.config.output_dir) / run_name

        wandb_disabled = _is_truthy(os.getenv("WANDB_DISABLED"))
        wandb_has_key = bool(os.getenv("WANDB_API_KEY"))
        wandb_enabled = self.config.wandb.enabled and not wandb_disabled and wandb_has_key
        if self.config.wandb.enabled and not wandb_enabled:
            logger.warning(
                "wandb_disabled_runtime",
                reason="missing WANDB_API_KEY or WANDB_DISABLED set",
            )

        # Configure WandB.
        if wandb_enabled:
            os.environ.setdefault("WANDB_PROJECT", self.config.wandb.project)

        effective_fp16 = self.config.training.fp16
        effective_bf16 = self.config.training.bf16
        if not self._use_unsloth and (effective_fp16 or effective_bf16):
            # PEFT fallback can hit AMP scaler dtype issues on some V100/torch stacks.
            # Keep mixed precision enabled only on Ampere+ GPUs (compute capability >= 8.0).
            allow_mixed_precision = False
            device_capability = "unknown"
            try:
                import torch

                if torch.cuda.is_available():
                    major, minor = torch.cuda.get_device_capability(0)
                    device_capability = f"{major}.{minor}"
                    allow_mixed_precision = major >= 8
            except Exception:  # noqa: BLE001
                allow_mixed_precision = False

            if not allow_mixed_precision:
                logger.warning(
                    "mixed_precision_disabled_peft_fallback",
                    reason="prevent amp scaler dtype mismatch on non-Ampere fallback backend",
                    device_capability=device_capability,
                    requested_fp16=effective_fp16,
                    requested_bf16=effective_bf16,
                )
                effective_fp16 = False
                effective_bf16 = False
            else:
                logger.info(
                    "mixed_precision_enabled_peft_fallback",
                    device_capability=device_capability,
                    fp16=effective_fp16,
                    bf16=effective_bf16,
                )

        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            fp16=effective_fp16,
            bf16=effective_bf16,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            eval_strategy="steps",
            eval_steps=self.config.training.eval_steps,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler_type=self.config.training.lr_scheduler_type,
            weight_decay=self.config.training.weight_decay,
            seed=self.config.training.seed,
            max_steps=self.config.training.max_steps,
            report_to="wandb" if wandb_enabled else "none",
            run_name=run_name,
            max_length=self.config.model.max_seq_length,
            packing=False,
        )

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            formatting_func=self._format_prompt,
        )

        logger.info("training_started", output_dir=str(output_dir))
        if resume_from_checkpoint:
            logger.info("training_resuming", checkpoint=resume_from_checkpoint)
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save final adapter
        final_path = output_dir / "final"
        self.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        logger.info("training_complete", adapter_path=str(final_path))

        return final_path
