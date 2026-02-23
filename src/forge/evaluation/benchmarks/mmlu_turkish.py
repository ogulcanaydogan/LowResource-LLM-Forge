"""Turkish MMLU evaluation via lm-evaluation-harness."""

from __future__ import annotations

from typing import Any

from forge.utils.logging import get_logger

logger = get_logger(__name__)


class TurkishMMLUBenchmark:
    """Run Turkish MMLU using the lm_eval library.

    Uses the 'turkishmmlu' task from lm-evaluation-harness which contains
    Turkish translations of the standard MMLU benchmark.
    """

    TASK_NAME = "turkishmmlu"
    PASS_THRESHOLD = 0.40  # random baseline is 0.25

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.model_path = model_path
        self.device = device

    def run(self) -> dict[str, Any]:
        """Run Turkish MMLU evaluation.

        Returns dict with overall_accuracy, per_subject scores, and num_questions.
        """
        try:
            import lm_eval
        except ImportError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Turkish MMLU requires lm-evaluation-harness (`lm_eval`) which is not "
                "installed in this environment."
            ) from exc

        logger.info("running_turkish_mmlu", model=self.model_path)

        # NOTE: dtype=float16 mandatory, lm_eval defaults to fp32 which OOMs
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={self.model_path},dtype=float16",
            tasks=[self.TASK_NAME],
            device=self.device,
            batch_size="auto",
        )

        task_results = results.get("results", {}).get(self.TASK_NAME, {})
        # lm_eval >=0.4 changed the key format
        accuracy = task_results.get("acc,none", task_results.get("acc", 0.0))

        output = {
            "overall_accuracy": accuracy,
            "task": self.TASK_NAME,
            "raw_results": {
                k: v for k, v in task_results.items() if isinstance(v, (int, float))
            },
        }

        logger.info("turkish_mmlu_complete", accuracy=accuracy)
        return output
