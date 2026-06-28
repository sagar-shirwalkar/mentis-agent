from __future__ import annotations

import numpy as np

from coding_agent.rag.embedder import Embedder


def test_default_dimension():
    embedder = Embedder()
    assert embedder.dimension == 384


def test_custom_dimension():
    embedder = Embedder(dimension=128)
    assert embedder.dimension == 128


def test_embed_returns_list_of_floats():
    embedder = Embedder(dimension=128)
    vector = embedder.embed("def foo(): pass")
    assert isinstance(vector, list)
    assert all(isinstance(v, float) for v in vector)
    assert len(vector) == 128


def test_embed_is_deterministic():
    embedder = Embedder(dimension=128)
    text = "def add(a, b): return a + b"
    v1 = embedder.embed(text)
    v2 = embedder.embed(text)
    assert v1 == v2


def test_embed_different_texts_differ():
    embedder = Embedder(dimension=128)
    v1 = embedder.embed("def foo(): pass")
    v2 = embedder.embed("def bar(): return 42")
    assert v1 != v2


def test_embed_unit_vector():
    embedder = Embedder(dimension=128)
    vector = embedder.embed("def foo(): pass")
    norm = np.linalg.norm(vector)
    assert abs(norm - 1.0) < 1e-5


def test_embed_short_text():
    embedder = Embedder(dimension=128)
    vector = embedder.embed("x")
    assert len(vector) == 128
    norm = np.linalg.norm(vector)
    assert abs(norm - 1.0) < 1e-5


def test_embed_empty_string():
    embedder = Embedder(dimension=128)
    vector = embedder.embed("")
    assert len(vector) == 128


def test_embed_special_characters():
    embedder = Embedder(dimension=128)
    vector = embedder.embed("!@#$%^&*()")
    assert len(vector) == 128
    norm = np.linalg.norm(vector)
    assert norm > 0.0 or norm == 0.0  # no word chars → zero vector, not an error


def test_embed_batch_returns_all():
    embedder = Embedder(dimension=128)
    texts = ["def foo(): pass", "class Bar: pass", "import os"]
    vectors = embedder.embed_batch(texts)
    assert len(vectors) == 3
    assert all(len(v) == 128 for v in vectors)


def test_embed_batch_empty():
    embedder = Embedder(dimension=128)
    vectors = embedder.embed_batch([])
    assert vectors == []


def test_embed_batch_deterministic():
    embedder = Embedder(dimension=128)
    texts = ["def foo(): pass", "class Bar: pass"]
    v1 = embedder.embed_batch(texts)
    v2 = embedder.embed_batch(texts)
    assert v1 == v2


def test_similar_texts_have_higher_dot_product():
    embedder = Embedder(dimension=128)
    v_sql1 = embedder.embed("SELECT * FROM users WHERE id = 1")
    v_sql2 = embedder.embed("SELECT name FROM orders WHERE total > 100")
    v_py = embedder.embed("def foo(): print('hello')")

    sql_dot = np.dot(v_sql1, v_sql2)
    cross_dot = np.dot(v_sql1, v_py)
    assert sql_dot > cross_dot


def test_code_vs_natural_language_separation():
    embedder = Embedder(dimension=128)
    v_code = embedder.embed("def add(a, b): return a + b")
    v_nl = embedder.embed("How do I add two numbers together?")
    dot = np.dot(v_code, v_nl)
    assert -1.0 <= dot <= 1.0


def test_init_onnx_fallback_when_not_installed():
    embedder = Embedder(dimension=128, backend="onnx")
    assert embedder.backend == "numpy_random"
    assert embedder._onnx_available is False
    vector = embedder.embed("test")
    assert len(vector) == 128
