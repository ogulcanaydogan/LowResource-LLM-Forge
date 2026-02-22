"""Data preprocessor: cleaning, deduplication, and quality filtering."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from forge.utils.config import PreprocessingConfig
from forge.utils.logging import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Clean and deduplicate training data."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self._seen_hashes: set[str] = set()

    def process_file(self, input_path: Path, output_path: Path) -> dict[str, int]:
        """Process a JSONL file. Returns stats dict with counts."""
        stats = {"total": 0, "kept": 0, "too_short": 0, "too_long": 0, "duplicate": 0}

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, encoding="utf-8") as fin, open(
            output_path, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                stats["total"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                record = self._clean_record(record)
                text = self._extract_text(record)

                if len(text) < self.config.min_length:
                    stats["too_short"] += 1
                    continue

                if len(text) > self.config.max_length:
                    stats["too_long"] += 1
                    continue

                if self._is_duplicate(text):
                    stats["duplicate"] += 1
                    continue

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["kept"] += 1

        logger.info("preprocessing_complete", path=str(input_path), **stats)
        return stats

    def _clean_record(self, record: dict[str, str]) -> dict[str, str]:
        """Apply cleaning to all text fields in a record."""
        cleaned = {}
        for key, value in record.items():
            if isinstance(value, str):
                cleaned[key] = self._clean_text(value)
            else:
                cleaned[key] = value
        return cleaned

    def _clean_text(self, text: str) -> str:
        """Apply cleaning pipeline: HTML removal, unicode normalization, whitespace."""
        if self.config.clean_html:
            text = re.sub(r"<[^>]+>", "", text)

        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Remove control characters (keep newlines and tabs)
        text = "".join(
            c for c in text
            if c in ("\n", "\t") or not unicodedata.category(c).startswith("C")
        )

        return text

    def _extract_text(self, record: dict[str, str]) -> str:
        """Extract concatenated text from all relevant fields for length/dedup checks."""
        parts = []
        for key in ("instruction", "input", "output"):
            val = record.get(key, "")
            if isinstance(val, str) and val:
                parts.append(val)
        return " ".join(parts)

    def _is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate based on configured method."""
        if self.config.dedup_method == "exact":
            return self._exact_dedup(text)
        elif self.config.dedup_method == "minhash":
            # MinHash approximation using multiple hash seeds
            return self._minhash_dedup(text)
        return False

    def _exact_dedup(self, text: str) -> bool:
        """Exact deduplication using SHA-256 hash."""
        h = hashlib.sha256(text.encode()).hexdigest()
        if h in self._seen_hashes:
            return True
        self._seen_hashes.add(h)
        return False

    def _minhash_dedup(self, text: str) -> bool:
        """Approximate deduplication using MinHash-style n-gram hashing.

        Uses a simplified approach: hash character-level n-grams with multiple
        seeds and compare Jaccard similarity estimate against threshold.
        """
        ngram_size = 5
        num_hashes = 64

        if len(text) < ngram_size:
            return self._exact_dedup(text)

        ngrams = {text[i : i + ngram_size] for i in range(len(text) - ngram_size + 1)}

        # Compute signature: minimum hash for each seed
        signature = []
        for seed in range(num_hashes):
            min_hash = min(
                int(hashlib.md5(f"{seed}:{ng}".encode()).hexdigest()[:8], 16)
                for ng in ngrams
            )
            signature.append(min_hash)

        sig_key = ",".join(str(s) for s in signature)

        # Compare against all seen signatures using Jaccard estimate
        for seen_sig_key in self._seen_hashes:
            seen_parts = seen_sig_key.split(",")
            if len(seen_parts) != num_hashes:
                continue
            matches = sum(
                1
                for a, b in zip(
                    signature, (int(x) for x in seen_parts), strict=False
                )
                if a == b
            )
            similarity = matches / num_hashes
            if similarity >= self.config.dedup_threshold:
                return True

        self._seen_hashes.add(sig_key)
        return False

    def reset(self) -> None:
        """Clear deduplication state for a new processing run."""
        self._seen_hashes.clear()
