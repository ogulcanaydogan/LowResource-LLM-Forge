"""Tests for data collector normalization variants."""

from __future__ import annotations

from forge.data.collector import DataCollector
from forge.utils.config import DataConfig


def _collector() -> DataCollector:
    cfg = DataConfig(language="tr", language_name="Turkish", sources=[])
    return DataCollector(cfg)


def test_normalize_alpaca_standard() -> None:
    collector = _collector()
    record = {"instruction": "Soru", "input": "Bağlam", "output": "Cevap"}
    out = collector._normalize_record(record, "alpaca")
    assert out == {"instruction": "Soru", "input": "Bağlam", "output": "Cevap"}


def test_normalize_alpaca_turkish_columns() -> None:
    collector = _collector()
    record = {"talimat": "Talimat", " giriş": "Girdi", " çıktı": "Çıktı"}
    out = collector._normalize_record(record, "alpaca")
    assert out == {"instruction": "Talimat", "input": "Girdi", "output": "Çıktı"}


def test_normalize_alpaca_chat_triplet() -> None:
    collector = _collector()
    record = {"system": "Sistem", "user": "Kullanıcı sorusu", "assistant": "Asistan cevabı"}
    out = collector._normalize_record(record, "alpaca")
    assert out == {
        "instruction": "Kullanıcı sorusu",
        "input": "Sistem",
        "output": "Asistan cevabı",
    }


def test_normalize_sharegpt_messages() -> None:
    collector = _collector()
    record = {
        "messages": [
            {"role": "user", "content": "Merhaba"},
            {"role": "assistant", "content": "Selam!"},
        ]
    }
    out = collector._normalize_record(record, "sharegpt")
    assert out == {"instruction": "Merhaba", "input": "", "output": "Selam!"}

