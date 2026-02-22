"""Tests for the data preprocessor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.data.preprocessor import DataPreprocessor
from forge.utils.config import PreprocessingConfig


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    config = PreprocessingConfig(
        min_length=10,
        max_length=1000,
        dedup_method="exact",
        clean_html=True,
        normalize_unicode=True,
    )
    return DataPreprocessor(config)


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_clean_text_removes_html(preprocessor: DataPreprocessor) -> None:
    result = preprocessor._clean_text("Hello <b>world</b> <script>alert('x')</script>")
    assert "<b>" not in result
    assert "<script>" not in result
    assert "Hello" in result
    assert "world" in result


def test_clean_text_normalizes_whitespace(preprocessor: DataPreprocessor) -> None:
    result = preprocessor._clean_text("Hello    world   \t  test")
    assert result == "Hello world test"


def test_process_file_filters_short(preprocessor: DataPreprocessor, tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    records = [
        {"instruction": "Short", "input": "", "output": "Hi"},
        {
            "instruction": "This is a longer instruction that passes the min length",
            "input": "",
            "output": "A good response here",
        },
    ]
    _write_jsonl(input_path, records)

    stats = preprocessor.process_file(input_path, output_path)
    assert stats["total"] == 2
    assert stats["too_short"] == 1
    assert stats["kept"] == 1


def test_process_file_deduplicates(preprocessor: DataPreprocessor, tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    dup_record = {
        "instruction": "This is a test instruction for dedup",
        "input": "",
        "output": "Response one",
    }
    records = [
        dup_record,
        dup_record,
        {"instruction": "Different instruction here", "input": "", "output": "Response two"},
    ]
    _write_jsonl(input_path, records)

    stats = preprocessor.process_file(input_path, output_path)
    assert stats["kept"] == 2
    assert stats["duplicate"] == 1


def test_process_file_filters_long(tmp_path: Path) -> None:
    config = PreprocessingConfig(min_length=10, max_length=50, dedup_method="exact")
    preprocessor = DataPreprocessor(config)

    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    records = [
        {"instruction": "x" * 100, "input": "", "output": "y" * 100},
        {"instruction": "Short but valid instruction", "input": "", "output": "Ok"},
    ]
    _write_jsonl(input_path, records)

    stats = preprocessor.process_file(input_path, output_path)
    assert stats["too_long"] == 1


def test_reset_clears_dedup_state(preprocessor: DataPreprocessor) -> None:
    preprocessor._seen_hashes.add("test_hash")
    assert len(preprocessor._seen_hashes) == 1
    preprocessor.reset()
    assert len(preprocessor._seen_hashes) == 0
