"""Tests for Turkic language detection heuristic."""

from __future__ import annotations

from forge.data.preprocessor import detect_language_heuristic


def test_detect_turkish() -> None:
    text = "Turkiye'de ekonomi ve sanayi için daha fazla yatirim yapilmali"
    assert detect_language_heuristic(text) == "tr"


def test_detect_azerbaijani() -> None:
    text = "Bakıda iqtisadiyyat və sənaye üçün daha çox investisiya lazımdır"
    assert detect_language_heuristic(text) == "az"


def test_inconclusive_returns_none() -> None:
    """Short or ambiguous text with <2 marker matches returns None."""
    assert detect_language_heuristic("hello world") is None
    assert detect_language_heuristic("") is None


def test_filter_with_candidates() -> None:
    """Only checks candidates that are specified."""
    # "ve" and "için" are unique to Turkish markers.
    text = "Hayat ve sanayi için gelişiyor"
    # With only tr as candidate, detection should work.
    assert detect_language_heuristic(text, candidates={"tr"}) == "tr"
    # With only az as candidate, no az-specific markers match.
    assert detect_language_heuristic(text, candidates={"az"}) is None


def test_shared_words_resolve_by_count() -> None:
    """When both languages share words, the one with more unique hits wins."""
    # "bir" is shared. "və" and "ilə" are unique to az.
    text = "bir və ilə yeni layihə başladı"
    assert detect_language_heuristic(text) == "az"
