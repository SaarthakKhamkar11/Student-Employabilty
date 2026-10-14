"""The currently configured embedding model, exposed as a provider the generate_embedding tool calls.

EMBEDDING_MODEL picks the provider; agents and tools stay as they are.
  e5_base_v2      E5, the semantic model selected by Phase 6 (docs/adr/0012): the primary model
  tfidf_baseline  the TF-IDF baseline (docs/adr/0004): the floor every model must beat, and the fallback

Fallback (docs/adr/0012). When E5 can't be loaded (weights or runtime missing), build_embedding_provider(...,
verify=True) returns the TF-IDF provider with `fallback_reason` set, and the system keeps working on it: TF-IDF
vectors are stored next to E5's for every profile and job (fallback_provider), and calibration, skill-gap tiers,
learning thresholds and the matching model all have TF-IDF entries.

Roles. Asymmetric models embed the two sides differently (E5: "query: " / "passage: "), so embed() takes a role,
fixed by the ENTITY, not by the matching direction, exactly as Phase 6 scored the models:
  QUERY    student texts (profile sections, stated interest, field) and skill names (symmetric: both sides query)
  PASSAGE  job texts (posting sections, title) and learning-resource texts
TF-IDF ignores the role.
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from app.core.config import Settings


class Vector(BaseModel):
    """One embedding. Sparse when `indices` is set (TF-IDF, ADR 0011), dense otherwise."""

    model_config = ConfigDict(extra="forbid")

    values: list[float]
    indices: list[int] | None = None  # 0-based positions of `values` in a vector of `dimension`


Role = Literal["query", "passage"]
QUERY: Role = "query"
PASSAGE: Role = "passage"


class EmbeddingProvider(Protocol):
    model_name: str
    interim: bool  # True while no model has been selected from experimental evidence

    def describe(self) -> tuple[str, int]: ...  # (model_version, dimension)

    def embed(self, texts: list[str], role: Role = PASSAGE) -> list[Vector | None]: ...  # None: no signal


class TfidfBaselineProvider:
    model_name = "tfidf_baseline"
    interim = True
    fallback_reason: str | None = None  # set when it stands in for an unavailable primary model

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

    def embed(self, texts: list[str], role: Role = PASSAGE) -> list[Vector | None]:
        rows = self._fitted().embed(texts)  # symmetric model: the role doesn't change a TF-IDF vector
        out: list[Vector | None] = []
        for row in rows:
            nonzero = np.flatnonzero(row)
            # A zero vector has no direction, so cosine is undefined: report it as "no signal", never store it.
            out.append(Vector(indices=nonzero.tolist(), values=row[nonzero].astype(float).tolist())
                       if nonzero.size else None)
        return out


class E5Provider:
    """E5 e5-base-v2 (intfloat), the model Phase 6 selected (docs/adr/0012), at the revision Phase 6 measured.

    Dense 768-d vectors, L2-normalised. Inputs are prefixed by role ("query: " / "passage: "), as in the Phase 6
    benchmark. Texts longer than the model's 512-token limit are truncated (Phase 6 reports the share). Results are
    cached per (role, text): job postings repeat their role's texts (376 distinct roles in 4,990 postings).
    """

    model_name = "e5_base_v2"
    interim = False
    HF_NAME = "intfloat/e5-base-v2"
    REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"  # = results/embedding_cost_measurements.json "e5"
    PREFIX = {QUERY: "query: ", PASSAGE: "passage: "}
    CACHE_SIZE = 50_000

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self._device, self._batch = device, batch_size
        self._model = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], Vector] = OrderedDict()

    def _loaded(self):
        with self._lock:
            if self._model is None:
                from huggingface_hub import snapshot_download
                from sentence_transformers import SentenceTransformer

                path = snapshot_download(self.HF_NAME, revision=self.REVISION, local_files_only=True)
                self._model = SentenceTransformer(path, device=self._device)
            return self._model

    def describe(self) -> tuple[str, int]:
        return f"{self.HF_NAME}@{self.REVISION[:12]}", 768  # the version string Phase 6 recorded

    def check(self) -> None:
        """Load the weights now; raises when they or the runtime are missing."""
        self._loaded()

    def embed(self, texts: list[str], role: Role = PASSAGE) -> list[Vector | None]:
        prefix = self.PREFIX[role]
        with self._lock:
            known = {t: self._cache[(role, t)] for t in texts if (role, t) in self._cache}
        todo = list(dict.fromkeys(t for t in texts if t and t.strip() and t not in known))
        if todo:
            rows = self._loaded().encode([prefix + t for t in todo], batch_size=self._batch,
                                         normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            fresh = {t: Vector(values=row.astype(float).tolist()) for t, row in zip(todo, rows)}
            known.update(fresh)
            with self._lock:
                self._cache.update({(role, t): v for t, v in fresh.items()})
                while len(self._cache) > self.CACHE_SIZE:
                    self._cache.popitem(last=False)
        return [known.get(t) if t and t.strip() else None for t in texts]


def build_embedding_provider(settings: Settings, verify: bool = False) -> EmbeddingProvider:
    """The configured model. verify=True loads it now and falls back to TF-IDF if it can't be loaded."""
    if settings.embedding_model == "e5_base_v2":
        e5 = E5Provider()
        if not verify:
            return e5
        try:
            e5.check()
            return e5
        except Exception as exc:  # noqa: BLE001 - any load failure means: run on the fallback
            from app.core.logging import get_logger, log_event

            fallback = TfidfBaselineProvider(settings.processed_data_dir)
            fallback.fallback_reason = f"{e5.model_name} unavailable ({type(exc).__name__})"
            log_event(get_logger("embeddings"), "embedding_fallback", logging.WARNING, primary=e5.model_name,
                      fallback=fallback.model_name, error=type(exc).__name__)
            return fallback
    if settings.embedding_model == "tfidf_baseline":
        return TfidfBaselineProvider(settings.processed_data_dir)
    raise ValueError(f"unknown EMBEDDING_MODEL {settings.embedding_model!r}")


def fallback_provider(primary: EmbeddingProvider, settings: Settings) -> EmbeddingProvider | None:
    """The TF-IDF provider whose vectors are stored next to the primary model's, so the fallback always has
    vectors for every profile and job. None when TF-IDF is already the model in use."""
    if primary.model_name == TfidfBaselineProvider.model_name:
        return None
    return TfidfBaselineProvider(settings.processed_data_dir)
