"""Cog Predictor for Replicate deployment.

This file is used by Replicate's Cog framework to serve the model.
Deploy with: cog push r8.im/ogulcanaydogan/<model-name>
"""

from __future__ import annotations

from typing import Any


def _get_predictor_class() -> type:
    """Lazy import to avoid requiring cog in development."""
    from cog import BasePredictor, Input

    class Predictor(BasePredictor):
        """Replicate-compatible predictor for Turkish LLM inference."""

        def setup(self) -> None:
            """Load model into memory on container start."""
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.model = AutoModelForCausalLM.from_pretrained(
                "model_weights",
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained("model_weights")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        def predict(
            self,
            prompt: str = Input(description="Input prompt in Turkish or English"),
            max_tokens: int = Input(
                description="Maximum tokens to generate", default=512, ge=1, le=4096
            ),
            temperature: float = Input(
                description="Sampling temperature", default=0.7, ge=0.0, le=2.0
            ),
            top_p: float = Input(
                description="Top-p (nucleus) sampling", default=0.9, ge=0.0, le=1.0
            ),
        ) -> str:
            """Generate text from prompt."""
            import torch

            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            return str(response.strip())

    return Predictor


# Only expose Predictor when cog is available
try:
    Predictor: Any = _get_predictor_class()
except ImportError:
    Predictor = None
