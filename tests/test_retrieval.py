"""Tests for retrieval helpers (chunking + cosine similarity)."""

import math

from aishell.retrieval import _chunk_text, _cosine


def test_cosine_identical_vectors():
    v = [1.0, 2.0, 3.0]
    assert math.isclose(_cosine(v, v), 1.0, rel_tol=1e-9)


def test_cosine_orthogonal():
    assert math.isclose(_cosine([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_chunk_text_short():
    chunks = list(_chunk_text("hello world"))
    assert chunks == ["hello world"]


def test_chunk_text_splits_long_input():
    text = "a" * 3000
    chunks = list(_chunk_text(text))
    assert len(chunks) > 1
    # Every chunk respects the configured maximum size.
    assert all(len(c) <= 1200 for c in chunks)


def test_chunk_text_skips_empty():
    chunks = list(_chunk_text("   \n  \n   "))
    assert chunks == []
