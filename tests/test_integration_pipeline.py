"""Integration test: end-to-end data pipeline without GPU dependencies."""

from __future__ import annotations

import json
from pathlib import Path

from forge.data.dataset_builder import DatasetBuilder
from forge.data.preprocessor import DataPreprocessor
from forge.utils.config import DataOutputConfig, PreprocessingConfig


def _write_raw_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def test_full_data_pipeline(tmp_path: Path) -> None:
    """Data flows from raw JSONL through preprocessing to train/eval split."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "output"

    # 1) Write synthetic raw data (20 records).
    records = [
        {
            "instruction": f"Soru {i}: Turkiye'nin ekonomisi hakkinda bilgi ver.",
            "input": "",
            "output": f"Cevap {i}: " + "Turkiye ekonomisi buyuyor. " * 5,
        }
        for i in range(20)
    ]
    raw_path = raw_dir / "test_data.jsonl"
    _write_raw_jsonl(raw_path, records)

    # 2) Preprocess.
    pp_config = PreprocessingConfig(
        min_length=30,
        max_length=5000,
        dedup_method="exact",
        clean_html=True,
        normalize_unicode=True,
    )
    preprocessor = DataPreprocessor(pp_config)
    processed_path = processed_dir / "clean.jsonl"
    stats = preprocessor.process_file(raw_path, processed_path)

    assert stats["total"] == 20
    assert stats["kept"] == 20  # All records should pass
    assert stats["too_short"] == 0
    assert stats["duplicate"] == 0
    assert processed_path.exists()

    # 3) Build SFT dataset with train/eval split.
    sft_path = output_dir / "sft.jsonl"
    eval_path = output_dir / "eval.jsonl"
    out_config = DataOutputConfig(
        sft_path=str(sft_path),
        eval_path=str(eval_path),
        eval_split_ratio=0.2,
    )
    builder = DatasetBuilder(out_config)
    build_stats = builder.build_sft_dataset([processed_path])

    assert build_stats["total"] == 20
    assert build_stats["train"] + build_stats["eval"] == 20
    assert build_stats["eval"] >= 1  # At least 1 eval sample
    assert sft_path.exists()
    assert eval_path.exists()

    # 4) Verify output records are valid JSON.
    with open(sft_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert "instruction" in rec
            assert "output" in rec


def test_pipeline_with_duplicates(tmp_path: Path) -> None:
    """Duplicate records are removed during preprocessing."""
    raw_path = tmp_path / "dupes.jsonl"
    dup_record = {
        "instruction": "Ayni soru tekrar tekrar soruluyor",
        "input": "",
        "output": "Ve ayni cevap tekrar tekrar veriliyor burada",
    }
    _write_raw_jsonl(raw_path, [dup_record] * 10)

    pp_config = PreprocessingConfig(
        min_length=10,
        max_length=5000,
        dedup_method="exact",
    )
    preprocessor = DataPreprocessor(pp_config)
    out_path = tmp_path / "clean.jsonl"
    stats = preprocessor.process_file(raw_path, out_path)

    assert stats["total"] == 10
    assert stats["kept"] == 1
    assert stats["duplicate"] == 9


def test_pipeline_filters_short_records(tmp_path: Path) -> None:
    """Records shorter than min_length are filtered out."""
    raw_path = tmp_path / "mixed.jsonl"
    records = [
        {"instruction": "OK", "input": "", "output": "Yes"},  # Too short
        {
            "instruction": "Long question about Turkish economy and culture",
            "input": "",
            "output": "A detailed answer about economics " * 3,
        },
    ]
    _write_raw_jsonl(raw_path, records)

    pp_config = PreprocessingConfig(min_length=50, max_length=5000, dedup_method="exact")
    preprocessor = DataPreprocessor(pp_config)
    out_path = tmp_path / "clean.jsonl"
    stats = preprocessor.process_file(raw_path, out_path)

    assert stats["too_short"] == 1
    assert stats["kept"] == 1
