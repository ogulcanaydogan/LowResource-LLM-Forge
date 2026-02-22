"""Data collector: downloads datasets from HuggingFace and normalizes formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from forge.utils.config import DataConfig, DataSourceConfig
from forge.utils.logging import get_logger

logger = get_logger(__name__)

# Mapping from format names to normalization functions
ALPACA_KEYS = {"instruction", "input", "output"}
SHAREGPT_KEYS = {"conversations"}


class DataCollector:
    """Download and unify datasets from multiple HuggingFace sources."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.raw_dir = Path("data/raw") / config.language
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def collect_all(self, limit: int = 0) -> list[Path]:
        """Download all configured sources. Returns list of output file paths."""
        output_paths: list[Path] = []

        for source in self.config.sources:
            logger.info(
                "downloading_dataset",
                repo=source.repo,
                split=source.split,
                format=source.format,
            )
            try:
                path = self._download_hf_dataset(source, limit=limit)
                output_paths.append(path)
                logger.info("dataset_downloaded", path=str(path))
            except Exception as e:
                logger.error("download_failed", repo=source.repo, error=str(e))

        return output_paths

    def _download_hf_dataset(self, source: DataSourceConfig, limit: int = 0) -> Path:
        """Download a single HuggingFace dataset and write as normalized JSONL."""
        safe_name = source.repo.replace("/", "_")
        output_path = self.raw_dir / f"{safe_name}.jsonl"

        ds = load_dataset(
            source.repo,
            name=source.subset,
            split=source.split,
            trust_remote_code=True,
        )

        if limit > 0:
            ds = ds.select(range(min(limit, len(ds))))

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for record in ds:
                normalized = self._normalize_record(record, source.format)
                if normalized:
                    f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    count += 1

        logger.info("records_written", count=count, path=str(output_path))
        return output_path

    def _normalize_record(
        self, record: dict[str, Any], fmt: str
    ) -> dict[str, Any] | None:
        """Convert any supported format to unified alpaca-style dict.

        Output format: {"instruction": str, "input": str, "output": str}
        """
        try:
            if fmt == "alpaca":
                return self._from_alpaca(record)
            elif fmt == "sharegpt":
                return self._from_sharegpt(record)
            elif fmt == "raw_text":
                return self._from_raw_text(record)
            elif fmt == "dpo":
                return self._from_dpo(record)
            else:
                logger.warning("unknown_format", format=fmt)
                return None
        except (KeyError, TypeError) as e:
            logger.debug("normalization_failed", error=str(e))
            return None

    def _from_alpaca(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize alpaca format (instruction/input/output)."""
        instruction = record.get("instruction", "").strip()
        if not instruction:
            return None
        return {
            "instruction": instruction,
            "input": record.get("input", "").strip(),
            "output": record.get("output", "").strip(),
        }

    def _from_sharegpt(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize ShareGPT format (conversations list)."""
        convos = record.get("conversations", [])
        if len(convos) < 2:
            return None

        human_msg = ""
        assistant_msg = ""
        for turn in convos:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role in ("human", "user") and not human_msg:
                human_msg = content.strip()
            elif role in ("gpt", "assistant") and not assistant_msg:
                assistant_msg = content.strip()

        if not human_msg or not assistant_msg:
            return None

        return {
            "instruction": human_msg,
            "input": "",
            "output": assistant_msg,
        }

    def _from_raw_text(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize raw text (just a text field)."""
        text = record.get("text", "").strip()
        if not text:
            return None
        return {
            "instruction": "Continue the following text:",
            "input": text[:200],
            "output": text[200:] if len(text) > 200 else text,
        }

    def _from_dpo(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Normalize DPO format (prompt/chosen/rejected) to SFT using chosen."""
        prompt = record.get("prompt", "").strip()
        chosen = record.get("chosen", "").strip()
        if not prompt or not chosen:
            return None
        return {
            "instruction": prompt,
            "input": "",
            "output": chosen,
        }
