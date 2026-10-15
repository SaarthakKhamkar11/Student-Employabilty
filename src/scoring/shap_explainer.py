"""Phase 11: deterministic feature contributions for the active matching model (docs/adr/0019).

The Explainability Agent calls this module through its tools. The LLM never computes a contribution (spec §4,
rule 5). Which explainer is used is decided per model, never assumed:

    lightgbm, xgboost, decision_tree, random_forest, gradient_boosting   shap.TreeExplainer (exact TreeSHAP)
    logistic_regression                                                   shap.LinearExplainer (exact, log-odds)
    weighted_baseline_v1 (Baseline 3)    its own additive decomposition: weight x normalised value / weight covered
                                         (exact for that model; not SHAP, and labelled as such)

The model is loaded exactly as Stage 4 loads it (app/matching/predictor.py: config flag, manifest SHA-256 check),
so the explained model IS the predicting model. Two separate entry points:

    explain_pair(model, features)      one (student, job) prediction: a contribution per feature, additivity checked
    global_explanation(model)          the whole model: importance ranking, distributions, summary-plot data

    python -m src.scoring.shap_explainer global [--embedding-model e5_base_v2] [--model lightgbm] [--out <json>]
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.matching.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "explainability.yaml"
FEATURES_DIR = ROOT / "data" / "benchmark" / "matching_features"
RESULTS_DIR = ROOT / "results" / "explainability"
BASELINE = "weighted_baseline_v1"
ADDITIVITY_TOLERANCE = 1e-6
# features-v1 describes skills only as counts and ratios; under a tree ensemble their contributions are small and
# can be negative (a tree's output moves in steps; docs/adr/0015, 0016). Reported with that caveat, never as precise.
SKILL_FEATURES = ("skill_match_score", "required_skill_match_score", "preferred_skill_match_score",
                  "matched_skill_count", "missing_skill_count", "required_skill_count", "matched_skill_ratio",
                  "preferred_skill_ratio", "transferable_skill_score")


class ExplanationUnavailable(ValueError):
    """Contributions can't be computed faithfully for this model or input (never approximated instead)."""


@lru_cache(maxsize=1)
def explainability_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class ExplainedModel:
    """The Stage 4 predictor plus its explainer. `predictor` is a BaselinePredictor or a TrainedPredictor."""

    predictor: object
    embedding_model: str
    _explainer: object = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return self.predictor.name

    @property
    def version(self) -> str:
        return self.predictor.version

    @property
    def is_baseline(self) -> bool:
        return self.key == BASELINE

    @property
    def output_space(self) -> str:
        if self.is_baseline:
            return "suitability"
        return "probability" if self.key in ("decision_tree", "random_forest") else "log-odds"

    @property
    def explainer_name(self) -> str:
        if self.is_baseline:
            return ("built-in additive decomposition of Baseline 3 (weight x normalised value / weight covered); "
                    "exact for that model, not SHAP")
        return self.explainer().name

    def explainer(self):
        if self.is_baseline:
            raise ExplanationUnavailable("Baseline 3 has no SHAP explainer: its contributions are built in")
        if self._explainer is None:
            from src.scoring.matching_models import Explainer

            model = self.predictor.model()  # SHA-256 checked against the manifest before deserialising
            background = (background_rows(self.predictor.manifest) if model.fmt == "joblib"
                          else pd.DataFrame(columns=list(FEATURE_NAMES), dtype=float))  # TreeSHAP needs none
            self._explainer = Explainer(model, background)
        return self._explainer


def load_model(embedding_model: str, override: str | None = None) -> ExplainedModel:
    """The model Stage 4 uses for `embedding_model` (config/matching_model.yaml, its fallback block, or an
    explicit override such as the MATCHING_MODEL setting)."""
    from app.matching.config import get_matching_config
    from app.matching.predictor import build_predictor

    return ExplainedModel(build_predictor(get_matching_config(), override, embedding_model=embedding_model),
                          embedding_model)


def _frame(features: dict[str, float | None]) -> pd.DataFrame:
    unknown = sorted(set(features) - set(FEATURE_NAMES))
    if unknown:
        raise ExplanationUnavailable(f"not in the feature contract: {unknown}")
    row = {f: (math.nan if features.get(f) is None else float(features[f])) for f in FEATURE_NAMES}
    return pd.DataFrame([row], columns=list(FEATURE_NAMES), dtype=float)


# ---------------------------------------------------------------- one prediction

def explain_pair(model: ExplainedModel, features: dict[str, float | None]) -> dict:
    """Contributions for ONE prediction: base + sum(contributions) == the model's output (checked, in
    output_space), and the recomputed suitability is returned so the caller can compare it with the prediction
    it is explaining."""
    if set(features) != set(FEATURE_NAMES):
        raise ExplanationUnavailable(f"expected the {len(FEATURE_NAMES)} contract features, got {len(features)}")
    if model.is_baseline:
        from app.matching.baseline import predict

        out = predict(features, model.predictor.cfg)
        if out["suitability"] is None:
            raise ExplanationUnavailable("Baseline 3 could not score this pair (no weighted feature has evidence)")
        values = {f: float(out["contributions"].get(f, 0.0)) for f in FEATURE_NAMES}
        base, output = 0.0, float(out["suitability"])
        suitability = output
    else:
        frame = _frame(features)
        values_arr, base = model.explainer().explain(frame)
        loaded = model.predictor.model()
        output = float(loaded.margin(frame)[0])
        suitability = float(loaded.score(frame)[0])
        values = {f: float(v) for f, v in zip(FEATURE_NAMES, values_arr[0])}
    error = abs(sum(values.values()) + base - output)
    if error > ADDITIVITY_TOLERANCE:  # contributions that don't add up would be invented evidence
        raise ExplanationUnavailable(f"contributions do not add up to the model output (error {error:.2e})")
    return {"matching_model": model.key, "matching_model_version": model.version, "explainer": model.explainer_name,
            "output_space": model.output_space, "base_value": round(float(base), 8),
            "model_output": round(output, 8), "suitability": round(suitability, 6),
            "values": {f: round(v, 8) for f, v in values.items()}, "additivity_error": float(error)}


# ---------------------------------------------------------------- the whole model

def feature_matrix_path(manifest: dict) -> Path:
    return FEATURES_DIR / manifest["feature_matrix"]["file"]


@lru_cache(maxsize=4)
def _read_matrix(path: str, expected_sha256: str) -> pd.DataFrame:
    from src.scoring.matching_models import read_features, sha256

    actual = sha256(Path(path))
    if actual != expected_sha256:
        raise ExplanationUnavailable(f"{Path(path).name}: SHA-256 {actual[:12]} is not the one the model was "
                                     f"trained with ({expected_sha256[:12]})")
    return read_features(Path(path))


def matrix_for(manifest: dict) -> pd.DataFrame:
    path = feature_matrix_path(manifest)
    if not path.exists():
        raise ExplanationUnavailable(f"the model's feature matrix is not present: {path.name}")
    return _read_matrix(str(path), manifest["feature_matrix"]["sha256"])


def background_rows(manifest: dict) -> pd.DataFrame:
    """The same seeded 200-row train-split sample Phase 8 explained each model with."""
    cfg = explainability_config()["global"]
    frame = matrix_for(manifest)
    train = frame[frame["split"] == "train"].loc[:, list(FEATURE_NAMES)].reset_index(drop=True)
    rng = np.random.default_rng(int(cfg["seed"]))
    return train.iloc[np.sort(rng.choice(len(train), min(int(cfg["background_rows"]), len(train)), replace=False))]


def _round(values, digits: int = 6) -> list:
    return [None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), digits) for v in values]


def global_explanation(model: ExplainedModel, split: str | None = None) -> dict:
    """Feature importance (mean |contribution|), per-feature contribution distributions over every row of the
    split, and a fixed sample of (feature value, contribution) points for a summary plot, as data. A trained model
    only: Baseline 3's global structure is its weights (config/matching_baseline_v1.yaml)."""
    if model.is_baseline:
        raise ExplanationUnavailable("Baseline 3 is a hand-weighted sum: its global structure is its configured "
                                     "weights (config/matching_baseline_v1.yaml), not a SHAP distribution")
    cfg = explainability_config()["global"]
    split = split or cfg["split"]
    manifest = model.predictor.manifest
    frame = matrix_for(manifest)
    rows = frame[frame["split"] == split].reset_index(drop=True)
    if rows.empty:
        raise ExplanationUnavailable(f"no {split} rows in {manifest['feature_matrix']['file']}")
    X = rows.loc[:, list(FEATURE_NAMES)].astype(float)
    values, base = model.explainer().explain(X)
    margin = model.predictor.model().margin(X)
    additivity = float(np.max(np.abs(values.sum(axis=1) + base - margin)))
    if additivity > ADDITIVITY_TOLERANCE:
        raise ExplanationUnavailable(f"contributions do not add up to the model output (max error {additivity:.2e})")

    mean_abs = np.abs(values).mean(axis=0)
    order = sorted(range(len(FEATURE_NAMES)), key=lambda i: (-round(float(mean_abs[i]), 8), FEATURE_NAMES[i]))
    ranking, rank = [], 0
    for pos, i in enumerate(order):
        value = round(float(mean_abs[i]), 6)
        if pos == 0 or value != ranking[-1]["mean_abs_shap"]:
            rank = pos + 1  # equal importances share a rank: a tie is not a ranking
        ranking.append({"rank": rank, "feature": FEATURE_NAMES[i], "mean_abs_shap": value,
                        "used_by_model": bool(np.any(values[:, i] != 0)),
                        "skill_feature": FEATURE_NAMES[i] in SKILL_FEATURES})
    tied = {r["rank"] for r in ranking if sum(1 for o in ranking if o["rank"] == r["rank"]) > 1}
    for r in ranking:
        r["tied"] = r["rank"] in tied

    distributions = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = values[:, i]
        counts, edges = np.histogram(col, bins=int(cfg["histogram_bins"]))
        missing = X[name].isna().to_numpy()
        distributions[name] = {
            "mean": round(float(col.mean()), 6), "std": round(float(col.std(ddof=0)), 6),
            "mean_abs": round(float(np.abs(col).mean()), 6), "min": round(float(col.min()), 6),
            "max": round(float(col.max()), 6),
            "p05": round(float(np.percentile(col, 5)), 6), "p50": round(float(np.percentile(col, 50)), 6),
            "p95": round(float(np.percentile(col, 95)), 6), "share_positive": round(float((col > 0).mean()), 4),
            "share_negative": round(float((col < 0).mean()), 4), "rows_missing_value": int(missing.sum()),
            "mean_when_missing": round(float(col[missing].mean()), 6) if missing.any() else None,
            "histogram": {"edges": _round(edges), "counts": counts.astype(int).tolist()},
        }

    rng = np.random.default_rng(int(cfg["seed"]))
    sample = np.sort(rng.choice(len(X), min(int(cfg["summary_sample"]), len(X)), replace=False))
    summary_plot = {
        "rows": int(len(sample)), "sampling": f"seeded uniform sample (seed {cfg['seed']}) of the {split} split",
        "features": {name: {"feature_value": _round(X[name].to_numpy()[sample]),
                            "shap": _round(values[sample, i])} for i, name in enumerate(FEATURE_NAMES)},
        "pair_ids": (rows["student_id"].to_numpy()[sample] + "|" + rows["job_id"].to_numpy()[sample]).tolist(),
    }
    return {
        "kind": "global", "matching_model": model.key, "matching_model_version": model.version,
        "artifacts_experiment": manifest["experiment_id"], "embedding_model": manifest.get("embedding_model"),
        "embedding_version": manifest["embedding_version"], "feature_version": manifest["feature_version"],
        "feature_matrix": manifest["feature_matrix"]["file"], "explainer": model.explainer_name,
        "output_space": model.output_space, "base_value": round(float(base), 6),
        "split": split, "rows": int(len(X)), "additivity_max_abs_error": additivity,
        "importance_ranking": ranking, "distributions": distributions, "summary_plot": summary_plot,
        "notes": [
            "Importance = mean |SHAP| over every row of the split, in the model's output space "
            f"({model.output_space}); equal values share a rank and are not a ranking.",
            "Skill features (counts and ratios in features-v1) have small, sometimes negative contributions under a "
            "tree ensemble, whose output moves in steps (docs/adr/0015, 0016): read them as approximate.",
            "The model was trained on weak (rule-based) labels, so these importances describe what it learned from "
            "those rules (e.g. branch_match), not necessarily what makes a real fit (docs/adr/0015).",
        ],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Global SHAP explanation of the active matching model")
    ap.add_argument("command", choices=["global"])
    ap.add_argument("--embedding-model", default=None, help="default: config/matching_model.yaml embedding_model")
    ap.add_argument("--model", default=None, help="override the configured matching model (demo/tests only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    from app.matching.predictor import ACTIVE_PATH

    embedding = args.embedding_model or yaml.safe_load(ACTIVE_PATH.read_text(encoding="utf-8"))["embedding_model"]
    result = global_explanation(load_model(embedding, args.model))
    out = Path(args.out) if args.out else RESULTS_DIR / f"global-{result['matching_model_version']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{result['matching_model_version']} ({result['explainer']}), {result['rows']} {result['split']} rows, "
          f"additivity error {result['additivity_max_abs_error']:.1e}, {result['output_space']}", file=sys.stderr)
    for r in result["importance_ranking"][:10]:
        print(f"{r['rank']:>2}. {r['feature']:<28} {r['mean_abs_shap']:.4f}{'  (tied)' if r['tied'] else ''}")
    print(f"-> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
