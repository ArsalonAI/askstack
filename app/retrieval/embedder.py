"""Local sentence-transformers embedder — TRD §3.1, ADR 2.

This module is the swap point. Nothing outside it may import
`sentence_transformers`; everything else depends on the `Embedder` protocol in
`app/interfaces.py`, so moving to a hosted embedding API later is a one-file
change rather than a refactor.

Local and deterministic on purpose: retrieval metrics that need no API key are
reproducible in CI, and a recall@5 change is always a real change.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

import numpy as np

from app.config import settings

log = logging.getLogger(__name__)

# bge models are trained with an asymmetric query prefix. Applying it to
# passages, or omitting it from queries, silently costs several points of
# recall — which is why the protocol has two methods rather than a flag.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 64


class BGEEmbedder:
    """`Embedder` over a local sentence-transformers model."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or settings.embedding_model
        self._model = None
        self._lock = threading.Lock()
        # Inference is serialised, not just loading. §2.3 called the model
        # "thread-safe for inference"; it is not. Concurrent `encode` calls
        # from several `asyncio.to_thread` workers — which is exactly what one
        # request does, since the dense arm and the tool selector both embed,
        # times the concurrent sessions §13 requires — segfault the process.
        # A crash, not an exception: nothing to catch, nothing in the logs.
        # Encoding a short query is single-digit milliseconds, so the lock
        # costs far less than the concurrency it protects.
        self._inference_lock = threading.Lock()

    @property
    def model(self):
        # Loaded lazily and once: ~130 MB resident, and importing torch costs
        # seconds that every test which never embeds anything shouldn't pay.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("loading embedding model %s", self.model_id)
                    self._model = SentenceTransformer(self.model_id)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalized. No query prefix — these are passages."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self.model  # load outside the inference lock
        with self._inference_lock:
            vectors = model.encode(
                list(texts),
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vectors.astype(np.float32, copy=False)

    def embed_query(self, text: str) -> np.ndarray:
        """(dim,) float32, with the bge query prefix applied."""
        return self.embed([QUERY_PREFIX + text])[0]


_default: BGEEmbedder | None = None


def get_embedder() -> BGEEmbedder:
    """Process-wide singleton (§2.3)."""
    global _default
    if _default is None:
        _default = BGEEmbedder()
    return _default
