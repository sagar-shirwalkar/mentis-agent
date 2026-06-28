"""
Embedder: generates embedding vectors for code chunks.

Two backends:
  1. numpy_random (default) — deterministic hash-based embeddings,
     zero external dependencies. Useful for pipeline testing and
     prototyping.
  2. onnx (optional) — ONNX Runtime powered embeddings using
     all-MiniLM-L6-v2. Activated when onnxruntime is installed.

Both produce fixed-size vectors (384-dim for MiniLM, 128-dim for
numpy_random) for use with FAISS / numpy cosine search.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """
    Generates embedding vectors for code.

    Usage:
        embedder = Embedder(dimension=384, backend="numpy_random")
        vector = embedder.embed("def foo(): pass")
        vectors = embedder.embed_batch(["chunk1", "chunk2"])
    """

    def __init__(
        self,
        dimension: int = 384,
        backend: str = "numpy_random",
    ) -> None:
        self.dimension = dimension
        self.backend = backend
        self._onnx_available = False

        if backend == "onnx":
            self._init_onnx()

    def _init_onnx(self) -> None:
        """Try to initialise ONNX Runtime backend."""
        try:
            import onnxruntime  # type: ignore[import-not-found]  # noqa: F401

            self._onnx_available = True
            logger.info("ONNX Runtime available — using MiniLM embeddings")
        except ImportError:
            logger.warning(
                "ONNX Runtime not installed — falling back to numpy_random embeddings. "
                "Install with: uv pip install onnxruntime"
            )
            self.backend = "numpy_random"

    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Returns a fixed-size float vector.
        """
        if self.backend == "onnx" and self._onnx_available:
            return self._embed_onnx(text)
        return self._embed_random(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple texts in a batch.

        ONNX backend processes these together for efficiency.
        """
        if self.backend == "onnx" and self._onnx_available:
            return self._embed_batch_onnx(texts)
        return [self._embed_random(t) for t in texts]

    # ── numpy_random backend ──────────────────────────────────

    def _embed_random(self, text: str) -> list[float]:
        """
        Deterministic hash-based embedding.

        Uses feature hashing: each n-gram in the text contributes
        to a random projection. The result is consistent for the
        same input text (deterministic seed from content hash).

        This is not semantically meaningful — it's a placeholder
        until ONNX Runtime is available.
        """
        # Generate n-gram features
        words = re.findall(r"\w+", text.lower())
        features = np.zeros(self.dimension, dtype=np.float32)

        for word in words:
            word_seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            word_rng = np.random.default_rng(word_seed)
            projection = word_rng.standard_normal(self.dimension, dtype=np.float32)
            features += projection

        # Normalise to unit vector
        norm = np.linalg.norm(features)
        if norm > 0:
            features /= norm

        return features.tolist()

    # ── ONNX backend (optional) ───────────────────────────────

    def _embed_onnx(self, text: str) -> list[float]:
        """
        Embed using ONNX Runtime with all-MiniLM-L6-v2.

        Falls back to numpy_random if the model file is not found.
        """
        result = self._embed_batch_onnx([text])
        return result[0] if result else self._embed_random(text)

    def _embed_batch_onnx(self, texts: list[str]) -> list[list[float]]:
        """
        Batch embed using ONNX Runtime.

        Model file expected at a configurable path.
        """
        if not texts:
            return []

        try:
            import onnxruntime as ort

            model_path = self._find_model_path()
            if model_path is None:
                logger.warning("MiniLM ONNX model not found — using numpy_random")
                return [self._embed_random(t) for t in texts]

            session = ort.InferenceSession(str(model_path))
            input_name = session.get_inputs()[0].name

            # Simple tokenisation using split (real implementation would use tokenizers)
            inputs = self._tokenise_for_onnx(texts)
            outputs = session.run(None, {input_name: inputs})[0]

            return outputs.tolist()  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning("ONNX embedding failed: %s — using numpy_random", exc)
            return [self._embed_random(t) for t in texts]

    @staticmethod
    def _find_model_path() -> Any | None:
        """Find the MiniLM ONNX model file."""
        import os

        candidates = [
            os.environ.get("ONNX_MODEL_PATH"),
            os.path.expanduser("~/.cache/meredith/all-MiniLM-L6-v2.onnx"),
            "models/all-MiniLM-L6-v2.onnx",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    @staticmethod
    def _tokenise_for_onnx(texts: list[str]) -> np.ndarray:
        """
        Minimal tokenisation for ONNX MiniLM.

        Real implementation should use HuggingFace tokenizers.
        This is a placeholder that works for basic queries.
        """
        max_len = 128
        batch = np.zeros((len(texts), max_len), dtype=np.int64)

        for i, text in enumerate(texts):
            tokens = re.findall(r"\w+", text.lower())[: max_len - 2]
            # Simple hash-based token IDs
            ids = [(int(hashlib.md5(t.encode()).hexdigest()[:4], 16) % 30000) + 2 for t in tokens]
            batch[i, 0] = 101  # [CLS]
            batch[i, 1 : len(ids) + 1] = ids
            batch[i, len(ids) + 1] = 102  # [SEP]

        return batch
