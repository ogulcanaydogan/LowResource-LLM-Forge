"""Generation quality evaluation using sample Turkish prompts."""

from __future__ import annotations

from typing import Any

from forge.utils.logging import get_logger

logger = get_logger(__name__)

# Built-in Turkish evaluation prompts covering diverse topics
TURKISH_EVAL_PROMPTS = [
    "Turkiye'nin ekonomik durumu hakkinda kisa bir analiz yap.",
    "Yapay zekanin egitim sektorundeki etkilerini acikla.",
    "Istanbul'un tarihi ve kulturel onemini anlat.",
    "Iklim degisikliginin Turkiye uzerindeki etkilerini tartis.",
    "Saglikli beslenme icin gunluk onerilerde bulun.",
    "Turkiye'deki teknoloji girisimciliginin gelecegini degerlendir.",
    "Bir bilgisayar muhendisine kariyer tavsiyeleri ver.",
    "Turkce dil bilgisi kurallarini ozetle.",
    "Yenilenebilir enerji kaynaklarinin avantajlarini sirala.",
    "Uzaktan calismanin artilari ve eksilerini karsilastir.",
]


class GenerationQualityBenchmark:
    """Evaluate Turkish text generation quality.

    Generates responses to a set of Turkish prompts and scores them on:
    - Response length (non-trivial response)
    - Coherence (no repetition loops)
    - Turkish character usage (proper Turkish text)
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.device = device

    def run(self) -> dict[str, Any]:
        """Run generation evaluation. Returns scores and sample outputs."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("loading_model_for_generation", model=self.model_path)

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

        scores: list[float] = []
        samples: list[dict[str, str]] = []

        for prompt in TURKISH_EVAL_PROMPTS:
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )

            decoded = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            if isinstance(decoded, list):
                decoded = " ".join(decoded)
            response = decoded.strip()

            score = self._score_response(response)
            scores.append(score)
            samples.append({"prompt": prompt, "response": response[:500], "score": str(score)})

        avg_score = sum(scores) / len(scores) if scores else 0.0

        result = {
            "average_score": avg_score,
            "num_prompts": len(TURKISH_EVAL_PROMPTS),
            "scores": scores,
            "samples": samples[:3],  # Include first 3 samples for inspection
        }
        logger.info("generation_quality_complete", average_score=avg_score)
        return result

    def _score_response(self, response: str) -> float:
        """Score a response on a 0-5 scale based on heuristics.

        Criteria:
        - Length: at least 50 chars for a meaningful response
        - No repetition: detect repeating phrases
        - Turkish chars: presence of Turkish-specific characters
        """
        score = 0.0

        # Length score (0-2 points)
        if len(response) >= 200:
            score += 2.0
        elif len(response) >= 100:
            score += 1.5
        elif len(response) >= 50:
            score += 1.0
        elif len(response) >= 20:
            score += 0.5

        # Repetition check (0-1.5 points)
        words = response.split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.7:
                score += 1.5
            elif unique_ratio > 0.5:
                score += 1.0
            elif unique_ratio > 0.3:
                score += 0.5

        # Turkish character presence (0-1.5 points)
        turkish_chars = set("cCgGiIsSOoUu")  # crude but catches most Turkish text
        has_turkish = any(c in turkish_chars for c in response)
        if has_turkish:
            score += 1.0
        # Check for common Turkish words
        turkish_words = {"ve", "bir", "bu", "de", "da", "ile", "icin", "olan"}
        response_words = set(response.lower().split())
        turkish_overlap = len(response_words & turkish_words)
        if turkish_overlap >= 3:
            score += 0.5

        return min(score, 5.0)
