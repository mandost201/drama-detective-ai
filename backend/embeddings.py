"""Semantic embeddings via fastembed (ONNX, CPU-friendly, no torch).

If the model cannot load (e.g. no network on first run), the app keeps
working with keyword search only - this is reported honestly, never faked.
"""
import threading

import numpy as np

from .config import EMBED_MODEL

_lock = threading.Lock()
_model = None
_failed = False
_dim = None


def available() -> bool:
    return _load() is not None


def dim():
    _load()
    return _dim


def _load():
    global _model, _failed, _dim
    with _lock:
        if _model is not None or _failed:
            return _model
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=EMBED_MODEL)
            _dim = _model._model_dim if hasattr(_model, "_model_dim") else 384
            print(f"[embeddings] loaded {EMBED_MODEL} (dim={_dim})")
        except Exception as e:
            _failed = True
            print(f"[embeddings] unavailable: {e}")
        return _model


def embed(texts):
    """Embed a list of strings -> np.ndarray (n, dim), float32 normalized."""
    model = _load()
    if model is None:
        raise RuntimeError("embeddings unavailable")
    vecs = list(model.embed(texts))
    arr = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    return arr / norms
