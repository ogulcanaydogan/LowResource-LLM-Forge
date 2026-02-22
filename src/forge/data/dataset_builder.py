"""Build final SFT/DPO datasets from preprocessed data."""

from __future__ import annotations

import json
import random
from pathlib import Path

from forge.utils.config import DataOutputConfig
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """Assemble final training and evaluation datasets from preprocessed files."""

    def __init__(self, config: DataOutputConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed

    def build_sft_dataset(self, input_paths: list[Path]) -> dict[str, int]:
        """Merge, shuffle, and split preprocessed files into train/eval JSONL.

        Returns stats dict with sample counts.
        """
        records: list[dict[str, str]] = []

        for path in input_paths:
            if not path.exists():
                logger.warning("input_not_found", path=str(path))
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            logger.error("no_records_found")
            return {"total": 0, "train": 0, "eval": 0}

        random.seed(self.seed)
        random.shuffle(records)

        split_idx = max(1, int(len(records) * (1 - self.config.eval_split_ratio)))
        train_records = records[:split_idx]
        eval_records = records[split_idx:]

        sft_path = Path(self.config.sft_path)
        eval_path = Path(self.config.eval_path)
        sft_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_jsonl(sft_path, train_records)
        self._write_jsonl(eval_path, eval_records)

        stats = {
            "total": len(records),
            "train": len(train_records),
            "eval": len(eval_records),
        }
        logger.info("dataset_built", **stats)
        return stats

    def _write_jsonl(self, path: Path, records: list[dict[str, str]]) -> None:
        """Write records to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("file_written", path=str(path), count=len(records))

    def get_stats(self) -> dict[str, int]:
        """Return dataset statistics from existing output files."""
        stats: dict[str, int] = {}
        for name, path_str in [
            ("sft", self.config.sft_path),
            ("eval", self.config.eval_path),
        ]:
            path = Path(path_str)
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    stats[name] = sum(1 for _ in f)
            else:
                stats[name] = 0
        return stats
