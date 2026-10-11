"""Stage 4's model, chosen by a config flag, never by a name in code (docs/adr/0015).

config/matching_model.yaml `active` (or the MATCHING_MODEL setting, for one process) names either
  * weighted_baseline_v1: Baseline 3, structured feature matching (app/matching/baseline.py), registered by
    `python -m app.matching.evaluate --register`; or
  * a trained Phase 8 candidate: loaded from models/matching/<artifacts_experiment>/, whose manifest gives the
    file's SHA-256 (checked before loading), the feature version and the embedding version it was trained on.

Every predictor exposes the spec §24 identity (name, version, experiment_id) and refuses to run with a
feature or embedding version other than the one it was trained on.
"""

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from app.core.config import REPO_ROOT
from app.matching import baseline
from app.matching.config import MatchingConfig
from app.matching.feature_builder import FEATURE_VERSION, normalise
from app.matching.features import FEATURE_NAMES

ACTIVE_PATH = REPO_ROOT / "config" / "matching_model.yaml"
MODELS_DIR = REPO_ROOT / "models" / "matching"
BASELINE = "weighted_baseline_v1"
NONE = "none"


class PredictorUnavailable(ValueError):
    """The configured model can't produce predictions with complete, consistent metadata."""


@dataclass(frozen=True)
class ActiveChoice:
    active: str
    artifacts_experiment: str
    selection_experiment: str
    source: str  # "config" or "MATCHING_MODEL"


def active_choice(override: str | None = None, path: Path = ACTIVE_PATH) -> ActiveChoice:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ActiveChoice(active=override or raw["active"], artifacts_experiment=str(raw["artifacts_experiment"]),
                        selection_experiment=str(raw["selection_experiment"]),
                        source="MATCHING_MODEL" if override else "config")


class BaselinePredictor:
    """Baseline 3 (weighted_baseline_v1): kept for Phase 14's baseline comparison and as the default until a
    trained model is selected on human labels."""

    def __init__(self, cfg: MatchingConfig):
        self.cfg = cfg
        self.name, self.version = cfg.matching_model, cfg.matching_model_version
        self.method = baseline.METHOD
        self.weights = cfg.weights

    @property
    def experiment_id(self) -> str | None:
        return self.cfg.registration.get("experiment_id") if self.cfg.registered else None

    def check(self, embedding_version: str) -> None:
        if not self.cfg.registered:  # short: agent errors longer than MAX_LOGGED_TEXT are redacted
            raise PredictorUnavailable(f"{self.version} not registered; no predictions")

    def predict_many(self, vectors: dict[str, dict[str, float | None]]) -> dict[str, dict]:
        return {c: baseline.predict(f, self.cfg) for c, f in vectors.items()}


@dataclass
class TrainedPredictor:
    key: str
    manifest: dict
    folder: Path
    cfg: MatchingConfig
    experiment_id: str
    _model: object = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.key

    @property
    def artifact(self) -> dict:
        return self.manifest["models"][self.key]

    @property
    def version(self) -> str:
        return f"{self.key}+{self.artifact['sha256'][:12]}"

    @property
    def method(self) -> str:
        return (f"{self.key}: trained Phase 8 model (experiment {self.manifest['experiment_id']}), predicted "
                "probability that the job is suitable (relevance >= 2); class-balanced, so a ranking score rather "
                "than a calibrated probability. Per-feature contributions: Phase 11 (SHAP)")

    weights: dict = field(default_factory=dict)  # a trained model has no hand-set weights

    def check(self, embedding_version: str) -> None:
        if self.manifest["feature_version"] != FEATURE_VERSION:
            raise PredictorUnavailable(f"{self.version} trained on {self.manifest['feature_version']}, "
                                       f"features are {FEATURE_VERSION}")
        if self.manifest["embedding_version"] != embedding_version:
            raise PredictorUnavailable(f"{self.version} trained on {self.manifest['embedding_version']}, "
                                       f"embeddings are {embedding_version}")

    def model(self):
        if self._model is None:
            from src.scoring.matching_models import load

            a = self.artifact
            self._model = load(self.folder / a["file"], self.key, a["format"], a["sha256"])
        return self._model

    def predict_many(self, vectors: dict[str, dict[str, float | None]]) -> dict[str, dict]:
        import pandas as pd

        if not vectors:
            return {}
        ids = list(vectors)
        frame = pd.DataFrame([{f: (math.nan if vectors[c].get(f) is None else vectors[c][f]) for f in FEATURE_NAMES}
                              for c in ids], columns=list(FEATURE_NAMES), dtype=float)
        scores = self.model().score(frame)
        out = {}
        for c, s in zip(ids, scores):
            normalised = {f: normalise(f, v, self.cfg) for f, v in vectors[c].items()}
            present = [f for f in FEATURE_NAMES if vectors[c].get(f) is not None]
            suitability = round(float(s), 6)
            out[c] = {"suitability": suitability, "overall": round(suitability * 100, 1), "normalised": normalised,
                      "contributions": {}, "weight_covered": round(len(present) / len(FEATURE_NAMES), 6),
                      "missing_features": sorted(set(FEATURE_NAMES) - set(present)),
                      "dimensions": baseline.dimensions(normalised, self.cfg)}
        return out


@lru_cache(maxsize=8)
def _manifest(folder: str) -> dict:
    path = Path(folder) / "manifest.json"
    if not path.exists():
        raise PredictorUnavailable(f"no trained-model manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_predictor(cfg: MatchingConfig, override: str | None = None, active_path: Path = ACTIVE_PATH,
                    models_dir: Path = MODELS_DIR):
    choice = active_choice(override, active_path)
    if choice.active == BASELINE:
        return BaselinePredictor(cfg)
    if choice.artifacts_experiment in ("", NONE):
        raise PredictorUnavailable(f"matching model {choice.active!r} requested but no artifacts_experiment is set")
    folder = models_dir / choice.artifacts_experiment
    manifest = _manifest(str(folder))
    if choice.active not in manifest["models"]:
        raise PredictorUnavailable(f"{choice.active!r} is not among the trained models of "
                                   f"{choice.artifacts_experiment}: {sorted(manifest['models'])}")
    selected = choice.selection_experiment not in ("", NONE) and choice.source == "config"
    return TrainedPredictor(choice.active, manifest, folder, cfg,
                            choice.selection_experiment if selected else manifest["experiment_id"])

