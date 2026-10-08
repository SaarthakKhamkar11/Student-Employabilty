"""The currently configured embedding model, exposed as a provider the generate_embedding tool calls.

Until Phase 6 selects a model from evidence, the only provider is the interim
TF-IDF baseline (docs/adr/0004). Phase 6 adds providers and changes
EMBEDDING_MODEL; agents and tools stay as they are.
"""

import threading
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from app.core.config import Settings


class Vector(BaseModel):
    """One embedding. Sparse when `indices` is set (TF-IDF, ADR 0011), dense otherwise."""

    model_config = ConfigDict(extra="forbid")

    values: list[float]
    indices: list[int] | None = None  # 0-based positions of `values` in a vector of `dimension`


class EmbeddingProvider(Protocol):
    model_name: str
    interim: bool  # True while no model has been selected from experimental evidence

    def describe(self) -> tuple[str, int]: ...  # (model_version, dimension)

    def embed(self, texts: list[str]) -> list[Vector | None]: ...  # None: the text carries no signal


class TfidfBaselineProvider:
    model_name = "tfidf_baseline"
    interim = True

    def __init__(self, processed_dir: Path):
        self._dir = processed_dir
        self._model = None
        self._lock = threading.Lock()

    def _fitted(self):
        with self._lock:  # tools run in worker threads; fit once
            if self._model is None:
                from src.embeddings.tfidf_baseline import TfidfBaseline

                students, jobs = self._dir / "students_unified.csv", self._dir / "jobs.csv"
                if not (students.exists() and jobs.exists()):
                    raise FileNotFoundError(f"TF-IDF corpus not found in {self._dir}")
                self._model = TfidfBaseline.fit_from_processed(str(students), str(jobs))
            return self._model

    def describe(self) -> tuple[str, int]:
        model = self._fitted()
        return model.version, model.dimension

    def embed(self, texts: list[str]) -> list[Vector | None]:
        rows = self._fitted().embed(texts)
        out: list[Vector | None] = []
        for row in rows:
            nonzero = np.flatnonzero(row)
            # A zero vector has no direction, so cosine is undefined: report it as "no signal", never store it.
            out.append(Vector(indices=nonzero.tolist(), values=row[nonzero].astype(float).tolist())
                       if nonzero.size else None)
        return out


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_model == "tfidf_baseline":
        return TfidfBaselineProvider(settings.processed_data_dir)
    raise ValueError(f"unknown EMBEDDING_MODEL {settings.embedding_model!r}")
