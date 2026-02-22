"""Unsloth QLoRA training orchestrator for low-resource language fine-tuning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import TrainingArguments

from forge.utils.config import TrainingConfig
from forge.utils.logging import get_logger

logger = get_logger(__name__)


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

        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            lora_dropout=self.config.lora.dropout,
            target_modules=self.config.lora.target_modules,
            bias=self.config.lora.bias,
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

    def _format_prompts(self, examples: dict[str, list[str]]) -> list[str]:
        """Format a batch of examples for SFTTrainer."""
        prompts = []
        for i in range(len(examples["instruction"])):
            example = {
                "instruction": examples["instruction"][i],
                "input": examples.get("input", [""] * len(examples["instruction"]))[i],
                "output": examples["output"][i],
            }
            prompts.append(self._format_prompt(example))
        return prompts

    def train(self) -> Path:
        """Run training. Returns path to saved adapter."""
        from trl import SFTTrainer

        if self.model is None:
            raise RuntimeError("Call setup() before train()")

        train_dataset, eval_dataset = self._load_datasets()

        run_name = self.config.wandb.run_name or "forge-run"
        output_dir = Path(self.config.output_dir) / run_name

        # Configure WandB
        if self.config.wandb.enabled:
            os.environ.setdefault("WANDB_PROJECT", self.config.wandb.project)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            fp16=self.config.training.fp16,
            bf16=self.config.training.bf16,
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
            report_to="wandb" if self.config.wandb.enabled else "none",
            run_name=run_name,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            formatting_func=self._format_prompts,
            max_seq_length=self.config.model.max_seq_length,
            packing=True,
        )

        logger.info("training_started", output_dir=str(output_dir))
        trainer.train()

        # Save final adapter
        final_path = output_dir / "final"
        self.model.save_pretrained(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        logger.info("training_complete", adapter_path=str(final_path))

        return final_path
