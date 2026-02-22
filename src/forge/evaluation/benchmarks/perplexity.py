"""Perplexity evaluation on held-out Turkish text."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class PerplexityBenchmark:
    """Calculate perplexity on held-out text.

    Lower perplexity = better language modeling.
    Compare fine-tuned vs base model to measure improvement.
    """

    def __init__(
        self,
        model_path: str,
        eval_data_path: str = "data/processed/turkish_eval.jsonl",
        max_samples: int = 500,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.eval_data_path = eval_data_path
        self.max_samples = max_samples
        self.device = device

    def run(self) -> dict[str, Any]:
        """Calculate perplexity. Returns dict with perplexity and metadata."""
        import json

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("loading_model_for_perplexity", model=self.model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load eval texts
        texts: list[str] = []
        eval_path = Path(self.eval_data_path)
        if eval_path.exists():
            with open(eval_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        text = " ".join(
                            record.get(k, "")
                            for k in ("instruction", "input", "output")
                            if record.get(k)
                        )
                        if text.strip():
                            texts.append(text.strip())
                    except json.JSONDecodeError:
                        continue
                    if len(texts) >= self.max_samples:
                        break

        if not texts:
            logger.warning("no_eval_data", path=str(eval_path))
            return {"perplexity": float("inf"), "num_samples": 0}

        # Calculate perplexity
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                encodings = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(model.device)

                outputs = model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss.item()
                num_tokens = encodings["input_ids"].size(1)

                total_loss += loss * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        result = {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "num_samples": len(texts),
            "num_tokens": total_tokens,
        }
        logger.info("perplexity_complete", **result)
        return result
