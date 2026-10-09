"""Stage 4, prediction: weighted_baseline_v1, an INTERIM transparent weighted sum (docs/adr/0014).

    suitability = sum(w_f * n_f) / sum(w_f)   over features f with weight > 0 that have a value

n_f is the feature normalised to [0, 1] (feature_builder.normalise). A feature without evidence is left out
and the weights renormalised over the rest; the output lists which were left out and how much weight
was covered. The weights are placeholders from config/matching_baseline_v1.yaml until Phase 8 compares
trained models. This is not a trained model, and every output says so.

The employability score (spec §15) is the same computation shown per dimension: each of the 8 dimensions
is the weighted mean of its own features' normalised values, x 100; the overall score is suitability x 100.
It is never a raw similarity x 100.
"""

from app.matching.config import MatchingConfig
from app.matching.feature_builder import normalise
from app.schemas.agent_state import INSUFFICIENT_EVIDENCE

METHOD = ("weighted_baseline_v1: a transparent weighted sum of normalised features with PLACEHOLDER weights "
          "pending Phase 8's model selection (docs/adr/0014); not a trained model")


def predict(features: dict[str, float | None], cfg: MatchingConfig) -> dict:
    normalised = {f: normalise(f, v, cfg) for f, v in features.items()}
    used = {f: w for f, w in cfg.weights.items() if w > 0 and normalised[f] is not None}
    total = sum(used.values())
    missing = sorted(f for f, w in cfg.weights.items() if w > 0 and normalised[f] is None)
    if total == 0:
        return {"suitability": None, "overall": INSUFFICIENT_EVIDENCE, "normalised": normalised, "contributions": {},
                "weight_covered": 0.0, "missing_features": missing, "dimensions": dimensions(normalised, cfg)}
    contributions = {f: round(w * normalised[f] / total, 6) for f, w in used.items()}
    suitability = round(sum(contributions.values()), 6)
    return {"suitability": suitability, "overall": round(suitability * 100, 1), "normalised": normalised,
            "contributions": contributions, "weight_covered": round(total, 6), "missing_features": missing,
            "dimensions": dimensions(normalised, cfg)}


def dimensions(normalised: dict[str, float | None], cfg: MatchingConfig) -> dict[str, float | str]:
    out = {}
    for dim, feats in cfg.dimensions.items():
        present = [f for f in feats if normalised.get(f) is not None]
        weights = {f: cfg.weights[f] for f in present if cfg.weights[f] > 0} or {f: 1.0 for f in present}
        if not weights:
            out[dim] = INSUFFICIENT_EVIDENCE
            continue
        out[dim] = round(100 * sum(w * normalised[f] for f, w in weights.items()) / sum(weights.values()), 1)
    return out
