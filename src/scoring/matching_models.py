"""Phase 8 matching-model candidates: build, train, save, load, score and explain (docs/adr/0015).

Every candidate sees the same 20 features-v1 columns (app/matching/features.py FEATURE_NAMES, checked by the
protected-attribute guard), the same rows and the same target (relevance_label >= 2). Hyperparameters come
from config/matching_model_candidates.yaml and are never tuned.

Saved formats: scikit-learn pipelines with joblib; XGBoost as its own JSON format; LightGBM as its own text
format. load() checks the file's SHA-256 against the manifest BEFORE deserialising: a joblib file can run code.

    python -m src.scoring.matching_models measure --key xgboost --features <csv.gz> --labels <weak csv> --out <dir>
runs one candidate in its own process (the engine calls this once per model), so memory and time are isolated.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from app.matching.features import FEATURE_NAMES, PROTECTED_ATTRIBUTES, validate_feature_names

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_PATH = ROOT / "config" / "matching_model_candidates.yaml"
BASELINE_KEY = "weighted_baseline_v1"
FORMATS = {"scikit-learn": ("joblib", ".joblib"), "xgboost": ("xgboost-json", ".json"),
           "lightgbm": ("lightgbm-text", ".txt")}
PREPROCESSING = ("median_impute_with_indicator", "standard_scale")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- config

def load_candidates(path: Path = CANDIDATES_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key, spec in cfg["candidates"].items():
        if spec["library"] not in FORMATS:
            raise ValueError(f"{key}: unknown library {spec['library']!r}")
        unknown = set(spec["preprocessing"]) - set(PREPROCESSING)
        if unknown:
            raise ValueError(f"{key}: unknown preprocessing {sorted(unknown)}")
        unexplained = [p for p in spec["params"] if not str(spec.get("reasons", {}).get(p, "")).strip()]
        if unexplained:
            raise ValueError(f"{key}: every hyperparameter needs a one-line reason; missing {unexplained}")
        for rubric in ("interpretability", "maintainability"):
            if not (1 <= int(spec[rubric]["score"]) <= 5 and str(spec[rubric]["reason"]).strip()):
                raise ValueError(f"{key}: {rubric} needs a 1-5 score and a reason")
    cfg["sha256"] = sha256(path)
    return cfg


def complexity_points(spec: dict) -> tuple[int, list[str]]:
    """Rubric in config/matching_model_candidates.yaml."""
    reasons, points = [], 0
    if spec.get("library", "scikit-learn") != "scikit-learn":
        points += 2
        reasons.append(f"+2 runtime dependency ({spec['library']})")
    for step in spec.get("preprocessing", []):
        points += 1
        reasons.append(f"+1 {step}")
    n = len(spec.get("params", {}))
    if n:
        points += math.ceil(n / 2)
        reasons.append(f"+{math.ceil(n / 2)} for {n} fixed hyperparameters")
    if FORMATS.get(spec.get("library"), ("joblib",))[0] != "joblib":
        points += 1
        reasons.append("+1 own model format and loader")
    return points, reasons or ["no dependency, preprocessing, hyperparameter or model format of its own"]


# ---------------------------------------------------------------- data

def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The 20 contract features, in contract order, as floats (None/missing -> NaN). The guard runs on the
    columns actually handed to a model, and anything protected in the frame is refused outright."""
    protected = [c for c in frame.columns if c.lower() in PROTECTED_ATTRIBUTES or "gender" in c.lower()]
    if protected:
        raise ValueError(f"protected attributes in the training data: {protected}")
    X = frame.loc[:, list(FEATURE_NAMES)].astype(float)
    validate_feature_names(X.columns)
    return X


def target(labels: pd.Series, threshold: int = 2) -> np.ndarray:
    return (labels.astype(int) >= threshold).astype(int).to_numpy()


def training_rows(features: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = cfg["training"]
    lab = labels[(labels["label_source"] == t["label_source"]) & (labels["split"] == t["split"])]
    rows = features.merge(lab[["student_id", "job_id", "relevance_label"]], on=["student_id", "job_id"], how="inner")
    if rows.empty:
        raise ValueError("no training rows: the feature matrix and the labels share no train pair")
    if set(rows["split"]) != {t["split"]}:
        raise ValueError("training rows outside the train split")
    return rows.sort_values(["student_id", "job_id"]).reset_index(drop=True)


# ---------------------------------------------------------------- models

def build_estimator(key: str, spec: dict, seed: int, threads: int, y: np.ndarray):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = dict(spec["params"])
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if spec["library"] == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**p, scale_pos_weight=n_neg / n_pos, n_jobs=threads, random_state=seed,
                             eval_metric="logloss")
    if spec["library"] == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**p, class_weight="balanced", n_jobs=threads, random_state=seed, verbose=-1)
    if key == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(**p, class_weight="balanced", random_state=seed)
    elif key == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        est = DecisionTreeClassifier(**p, class_weight="balanced", random_state=seed)
    elif key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        est = RandomForestClassifier(**p, class_weight="balanced", n_jobs=threads, random_state=seed)
    elif key == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier

        est = GradientBoostingClassifier(**p, random_state=seed)  # no class_weight: balanced via sample_weight
    else:
        raise ValueError(f"unknown scikit-learn candidate {key!r}")
    steps = []
    if "median_impute_with_indicator" in spec["preprocessing"]:
        steps.append(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    if "standard_scale" in spec["preprocessing"]:
        steps.append(StandardScaler())
    return make_pipeline(*steps, est)


def fit(key: str, spec: dict, X: pd.DataFrame, y: np.ndarray, seed: int, threads: int):
    model = build_estimator(key, spec, seed, threads, y)
    if key == "gradient_boosting":
        from sklearn.utils.class_weight import compute_sample_weight

        model.fit(X, y, gradientboostingclassifier__sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(X, y)
    return model


def save(model, key: str, spec: dict, folder: Path) -> tuple[Path, str]:
    fmt, ext = FORMATS[spec["library"]]
    path = Path(folder) / f"{key}{ext}"
    if fmt == "joblib":
        import joblib

        joblib.dump(model, path, compress=3)
    elif fmt == "xgboost-json":
        model.save_model(path)
    else:
        model.booster_.save_model(str(path))
    return path, fmt


@dataclass
class LoadedModel:
    key: str
    fmt: str
    obj: object

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted probability of 'suitable' (label >= 2). Class-balanced training: a ranking score, not a
        calibrated probability."""
        X = X.loc[:, list(FEATURE_NAMES)]
        if self.fmt == "lightgbm-text":
            return self.obj.predict(X.to_numpy(dtype=float))
        return self.obj.predict_proba(X)[:, 1]

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        """The model's raw output in the space its explanations add up in (log-odds, or probability for
        bagged trees)."""
        X = X.loc[:, list(FEATURE_NAMES)]
        if self.fmt == "lightgbm-text":
            return self.obj.predict(X.to_numpy(dtype=float), raw_score=True)
        if self.fmt == "xgboost-json":
            return self.obj.predict(X, output_margin=True)
        if self.key in ("decision_tree", "random_forest"):
            return self.obj.predict_proba(X)[:, 1]
        return self.obj.decision_function(X)


def load(path: Path, key: str, fmt: str, expected_sha256: str) -> LoadedModel:
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {actual[:12]} does not match the manifest ({expected_sha256[:12]}); "
                         "refusing to load")
    if fmt == "joblib":
        import joblib

        obj = joblib.load(path)
    elif fmt == "xgboost-json":
        from xgboost import XGBClassifier

        obj = XGBClassifier()
        obj.load_model(path)
    elif fmt == "lightgbm-text":
        import lightgbm

        obj = lightgbm.Booster(model_file=str(path))
    else:
        raise ValueError(f"unknown model format {fmt!r}")
    return LoadedModel(key, fmt, obj)


# ---------------------------------------------------------------- explanations

def _class1(values, expected):
    arr = np.asarray(values[1] if isinstance(values, list) else values, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, :, 1]
    base = np.asarray(expected, dtype=float).ravel()
    return arr, float(base[1] if base.size == 2 else base[0])


def _merge_indicators(values: np.ndarray, imputer) -> np.ndarray:
    """Columns after the imputer: the 20 features, then one indicator per feature that had missing values in
    training. A feature's contribution is its own column plus its indicator's."""
    n = len(FEATURE_NAMES)
    merged = values[:, :n].copy()
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None:
        for j, feature_index in enumerate(indicator.features_):
            merged[:, feature_index] += values[:, n + j]
    return merged


class Explainer:
    """Per-feature contributions (SHAP) for one loaded model, in the space margin() returns. The name of the
    explainer is recorded: compatibility is checked per model, never assumed."""

    def __init__(self, model: LoadedModel, background: pd.DataFrame):
        import shap

        self.model = model
        bg = background.loc[:, list(FEATURE_NAMES)]
        if model.fmt == "joblib":
            steps = model.obj.steps
            self.transform, self.final = steps[:-1], steps[-1][1]
            self.imputer = next((s for _, s in self.transform if s.__class__.__name__ == "SimpleImputer"), None)
            Xt = self._transform(bg)
            if model.key == "logistic_regression":
                self.name = "shap.LinearExplainer (exact for a linear model in log-odds, independent features)"
                self.explainer = shap.LinearExplainer(self.final, shap.maskers.Independent(Xt, max_samples=len(Xt)))
            else:
                self.name = "shap.TreeExplainer (exact TreeSHAP, tree_path_dependent)"
                self.explainer = shap.TreeExplainer(self.final)
        else:
            self.imputer, self.transform = None, []
            self.name = "shap.TreeExplainer (exact TreeSHAP, tree_path_dependent)"
            self.explainer = shap.TreeExplainer(model.obj)

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        Xt = X
        for _, step in self.transform:
            Xt = step.transform(Xt)
        return np.asarray(Xt, dtype=float)

    def explain(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        X = X.loc[:, list(FEATURE_NAMES)]
        if self.model.fmt == "joblib":
            values, base = _class1(self.explainer.shap_values(self._transform(X)), self.explainer.expected_value)
            return _merge_indicators(values, self.imputer) if self.imputer is not None else values, base
        data = X.to_numpy(dtype=float) if self.model.fmt == "lightgbm-text" else X
        return _class1(self.explainer.shap_values(data), self.explainer.expected_value)


# ---------------------------------------------------------------- isolated measurement (one process per model)

class PeakRSS:
    """Samples this process's resident memory every 5 ms; peak minus the level at start = memory added."""

    def __init__(self):
        self._stop = threading.Event()
        self.start_mb = self.peak_mb = rss_mb()

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb())
            time.sleep(0.005)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, rss_mb())

    @property
    def added_mb(self) -> float:
        return round(self.peak_mb - self.start_mb, 1)


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024**2


def _timed(fn, trials: int, warmup: int = 3) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return {"median": round(statistics.median(times), 4), "p95": round(float(np.percentile(times, 95)), 4),
            "trials": trials}


def _scalability(key, spec, X, y, seed, threads, full_seconds) -> dict:
    rng = np.random.default_rng(seed)
    points = []
    for share in (0.25, 0.5):
        idx = np.sort(rng.choice(len(X), int(len(X) * share), replace=False))
        t = time.perf_counter()
        fit(key, spec, X.iloc[idx], y[idx], seed, threads)
        points.append((len(idx), time.perf_counter() - t))
    points.append((len(X), full_seconds))
    n, s = np.log10([p[0] for p in points]), np.log10([max(p[1], 1e-6) for p in points])
    slope, intercept = np.polyfit(n, s, 1)
    extrapolate = lambda rows: round(float(10 ** (intercept + slope * math.log10(rows))), 1)  # noqa: E731
    return {"measured": [{"rows": r, "seconds": round(sec, 3)} for r, sec in points],
            "exponent": round(float(slope), 2),
            "training_seconds_1M_pairs": extrapolate(1_000_000), "training_seconds_10M_pairs": extrapolate(10_000_000),
            "basis": "EXTRAPOLATED: power law fitted to 3 measured training-set sizes (25%, 50%, 100% of the "
                     "train split), single thread; not measured at that scale"}


def measure_candidate(key: str, spec: dict, cfg: dict, features: pd.DataFrame, labels: pd.DataFrame,
                      out_dir: Path) -> dict:
    t_cfg = cfg["training"]
    seed, threads = int(t_cfg["seed"]), int(t_cfg["threads"])
    rows = training_rows(features, labels, cfg)
    X, y = feature_matrix(rows), target(rows["relevance_label"])
    held_out = features[features["split"] != "train"].reset_index(drop=True)
    Xh = feature_matrix(held_out)

    # Untimed warm-up fit on a sample: the first fit also imports the library's lazily loaded modules, which
    # would otherwise be counted as training memory and time.
    rng = np.random.default_rng(seed)
    warm = np.sort(rng.choice(len(X), min(5000, len(X)), replace=False))
    fit(key, spec, X.iloc[warm], y[warm], seed, threads)
    gc.collect()
    fits, mem = [], None
    for i in range(3):
        if i == 0:
            with PeakRSS() as mem:
                t = time.perf_counter()
                model = fit(key, spec, X, y, seed, threads)
                fits.append(time.perf_counter() - t)
        else:
            t = time.perf_counter()
            refit = fit(key, spec, X, y, seed, threads)
            fits.append(time.perf_counter() - t)
            if i == 1:
                diff = float(np.max(np.abs(refit.predict_proba(Xh)[:, 1] - model.predict_proba(Xh)[:, 1])))
    del refit
    path, fmt = save(model, key, spec, out_dir)
    digest = sha256(path)
    loaded = load(path, key, fmt, digest)
    scores = loaded.score(Xh)
    if not np.allclose(scores, model.predict_proba(Xh)[:, 1], atol=1e-6):
        raise RuntimeError(f"{key}: the saved artifact does not reproduce the trained model's scores")

    one, fifty = Xh.iloc[[0]], Xh.iloc[:50]
    throughput_runs = []
    for _ in range(5):
        t = time.perf_counter()
        loaded.score(Xh)
        throughput_runs.append(len(Xh) / (time.perf_counter() - t))

    background = X.iloc[np.sort(rng.choice(len(X), min(200, len(X)), replace=False))]
    check = Xh[held_out["split"] == "validation"].iloc[:200]
    explainer = Explainer(loaded, background)
    values, base = explainer.explain(check)
    additivity = float(np.max(np.abs(values.sum(axis=1) + base - loaded.margin(check))))
    importance = dict(sorted(zip(FEATURE_NAMES, np.abs(values).mean(axis=0).round(5).tolist()),
                             key=lambda kv: -kv[1]))
    shap_ms = _timed(lambda: explainer.explain(one), 10, warmup=2)

    return {
        "key": key, "display": spec["display"], "library": spec["library"], "params": spec["params"],
        "artifact": {"file": path.name, "format": fmt, "sha256": digest, "size_bytes": path.stat().st_size},
        "training": {"rows": int(len(X)), "positives": int(y.sum()), "seconds": [round(s, 3) for s in fits],
                     "median_seconds": round(statistics.median(fits), 3), "memory_added_mb": mem.added_mb,
                     "memory_peak_mb": round(mem.peak_mb, 1), "threads": threads, "seed": seed},
        "reproducibility": {"seed": seed, "max_abs_diff_between_refits": diff, "deterministic": diff == 0.0},
        "inference": {"single_pair_ms": _timed(lambda: loaded.score(one), 30),
                      "batch_50_ms": _timed(lambda: loaded.score(fifty), 30),
                      "throughput_pairs_per_s": round(statistics.median(throughput_runs), 1),
                      "throughput_rows": int(len(Xh))},
        "scalability": _scalability(key, spec, X, y, seed, threads, statistics.median(fits)),
        "explainability": {"explainer": explainer.name, "pairs_checked": int(len(check)),
                           "additivity_max_abs_error": additivity, "output_space": "probability" if key in (
                               "decision_tree", "random_forest") else "log-odds",
                           "single_pair_explanation_ms": shap_ms, "mean_abs_contribution": importance},
        "scores": dict(zip(held_out["student_id"] + "|" + held_out["job_id"], scores.round(8).tolist())),
    }


def measure_baseline(features: pd.DataFrame, out_dir: Path) -> dict:
    """Baseline 3 (weighted_baseline_v1) through the same measurements. Nothing is trained."""
    from app.matching.baseline import predict
    from app.matching.config import MATCHING_CONFIG_PATH, get_matching_config

    mcfg = get_matching_config()
    held_out = features[features["split"] != "train"].reset_index(drop=True)
    vectors = [{f: (None if pd.isna(v) else float(v)) for f, v in zip(FEATURE_NAMES, row)}
               for row in feature_matrix(held_out).itertuples(index=False)]
    preds = [predict(v, mcfg) for v in vectors]
    additivity = max(abs(sum(p["contributions"].values()) - p["suitability"]) for p in preds
                     if p["suitability"] is not None)
    throughput_runs = []
    for _ in range(5):
        t = time.perf_counter()
        [predict(v, mcfg) for v in vectors]
        throughput_runs.append(len(vectors) / (time.perf_counter() - t))
    return {
        "key": BASELINE_KEY, "display": "Baseline 3: structured feature matching", "library": "none",
        "params": {"weights": mcfg.weights},
        "artifact": {"file": MATCHING_CONFIG_PATH.name, "format": "yaml (weights)", "sha256": mcfg.sha256,
                     "size_bytes": MATCHING_CONFIG_PATH.stat().st_size,
                     "matching_model_version": mcfg.matching_model_version},
        "training": {"rows": 0, "positives": 0, "seconds": [0.0], "median_seconds": 0.0, "memory_added_mb": 0.0,
                     "memory_peak_mb": None, "threads": 1, "seed": None,
                     "note": "not trained: hand-set placeholder weights (docs/adr/0014)"},
        "reproducibility": {"seed": None, "max_abs_diff_between_refits": 0.0, "deterministic": True},
        "inference": {"single_pair_ms": _timed(lambda: predict(vectors[0], mcfg), 30),
                      "batch_50_ms": _timed(lambda: [predict(v, mcfg) for v in vectors[:50]], 30),
                      "throughput_pairs_per_s": round(statistics.median(throughput_runs), 1),
                      "throughput_rows": len(vectors)},
        "scalability": {"measured": [], "exponent": 0.0, "training_seconds_1M_pairs": 0.0,
                        "training_seconds_10M_pairs": 0.0, "basis": "nothing to train"},
        "explainability": {"explainer": "built in: weight x normalised feature / weight covered",
                           "pairs_checked": len(preds), "additivity_max_abs_error": additivity,
                           "output_space": "suitability",
                           "single_pair_explanation_ms": {"median": 0.0, "p95": 0.0, "trials": 0,
                                                          "note": "contributions come with every prediction"},
                           "mean_abs_contribution": {}},
        "scores": {f"{s}|{j}": p["suitability"] for s, j, p in zip(held_out["student_id"], held_out["job_id"], preds)
                   if p["suitability"] is not None},
    }


def read_features(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"student_id": str, "job_id": str, "split": str})
    feature_matrix(frame)  # refuses protected attributes / missing contract columns up front
    return frame


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Train and measure one matching-model candidate in this process")
    ap.add_argument("command", choices=["measure"])
    ap.add_argument("--key", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    features = read_features(Path(args.features))
    if args.key == BASELINE_KEY:
        result = measure_baseline(features, out)
    else:
        cfg = load_candidates()
        labels = pd.read_csv(args.labels, dtype=str)
        result = measure_candidate(args.key, cfg["candidates"][args.key], cfg, features, labels, out)
    (out / f"{args.key}.measurement.json").write_text(json.dumps(result), encoding="utf-8")
    print(f"{args.key}: measured", file=sys.stderr)


if __name__ == "__main__":
    main()
