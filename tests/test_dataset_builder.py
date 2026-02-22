"""Tests for the dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

from forge.data.dataset_builder import DatasetBuilder
from forge.utils.config import DataOutputConfig


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def test_build_sft_dataset(tmp_path: Path) -> None:
    config = DataOutputConfig(
        sft_path=str(tmp_path / "sft.jsonl"),
        eval_path=str(tmp_path / "eval.jsonl"),
        eval_split_ratio=0.2,
    )
    builder = DatasetBuilder(config, seed=42)

    input_path = tmp_path / "input.jsonl"
    records = [
        {"instruction": f"Instruction {i}", "input": "", "output": f"Output {i}"}
        for i in range(100)
    ]
    _write_jsonl(input_path, records)

    stats = builder.build_sft_dataset([input_path])

    assert stats["total"] == 100
    assert stats["train"] == 80
    assert stats["eval"] == 20

    train_records = _read_jsonl(Path(config.sft_path))
    eval_records = _read_jsonl(Path(config.eval_path))
    assert len(train_records) == 80
    assert len(eval_records) == 20


def test_build_sft_dataset_shuffles(tmp_path: Path) -> None:
    config = DataOutputConfig(
        sft_path=str(tmp_path / "sft.jsonl"),
        eval_path=str(tmp_path / "eval.jsonl"),
        eval_split_ratio=0.1,
    )
    builder = DatasetBuilder(config, seed=42)

    input_path = tmp_path / "input.jsonl"
    records = [
        {"instruction": f"Instruction {i}", "input": "", "output": f"Output {i}"}
        for i in range(50)
    ]
    _write_jsonl(input_path, records)

    builder.build_sft_dataset([input_path])

    train_records = _read_jsonl(Path(config.sft_path))
    # Verify shuffle happened (first record shouldn't be "Instruction 0" with seed=42)
    instructions = [r["instruction"] for r in train_records]
    assert instructions != [f"Instruction {i}" for i in range(len(instructions))]


def test_build_sft_dataset_multiple_inputs(tmp_path: Path) -> None:
    config = DataOutputConfig(
        sft_path=str(tmp_path / "sft.jsonl"),
        eval_path=str(tmp_path / "eval.jsonl"),
        eval_split_ratio=0.1,
    )
    builder = DatasetBuilder(config, seed=42)

    path1 = tmp_path / "source1.jsonl"
    path2 = tmp_path / "source2.jsonl"

    records_a = [{"instruction": f"A{i}", "input": "", "output": f"A{i}"} for i in range(30)]
    records_b = [{"instruction": f"B{i}", "input": "", "output": f"B{i}"} for i in range(20)]
    _write_jsonl(path1, records_a)
    _write_jsonl(path2, records_b)

    stats = builder.build_sft_dataset([path1, path2])
    assert stats["total"] == 50


def test_build_sft_dataset_empty_input(tmp_path: Path) -> None:
    config = DataOutputConfig(
        sft_path=str(tmp_path / "sft.jsonl"),
        eval_path=str(tmp_path / "eval.jsonl"),
    )
    builder = DatasetBuilder(config)

    stats = builder.build_sft_dataset([])
    assert stats["total"] == 0


def test_get_stats(tmp_path: Path) -> None:
    config = DataOutputConfig(
        sft_path=str(tmp_path / "sft.jsonl"),
        eval_path=str(tmp_path / "eval.jsonl"),
    )

    _write_jsonl(Path(config.sft_path), [{"a": "1"}, {"a": "2"}])

    builder = DatasetBuilder(config)
    stats = builder.get_stats()
    assert stats["sft"] == 2
    assert stats["eval"] == 0
