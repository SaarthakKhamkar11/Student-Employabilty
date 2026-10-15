"""Explainability Agent tools (spec §16, §20; docs/adr/0019): one tool per stage, validated input and output.

Individual match (one prediction):
    get_model_prediction       the MatchResult's prediction, feature vector and version metadata, checked
    get_shap_values            SHAP for that feature vector under the model that made the prediction
                               (src/scoring/shap_explainer.py); refuses another model or a score it can't reproduce
    get_feature_contributions  the pair's largest positive and negative contributions (config/explainability.yaml)
    generate_evidence          one EvidenceItem per claim the explanation may make
    generate_explanation       the LLM restates the evidence as lines citing evidence ids (no LLM -> no draft)
    validate_explanation       every line traced to the evidence it cites (app/explainability/validation.py)
    get_match_explanation      the final explanation: a validated draft, or the structured evidence; never an
                               unvalidated line
Global (the whole model):
    get_global_explanation     importance ranking, SHAP distributions over the validation split, summary-plot data
"""

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from app.core.config import Settings
from app.explainability import prose
from app.explainability.evidence import build_evidence
from app.explainability.validation import validate
from app.llm import LLMClient, traced
from app.llm.client import LLMError
from app.matching.features import FEATURE_NAMES
from app.schemas.agent_io import (
    EvidenceBundle, ExplanationValidation, GlobalExplanation, MatchExplanation, MatchResult, MatchTrace, ProseDraft,
    SkillGapReport,
)
from app.schemas.agent_state import PredictionVersions
from app.tools.registry import ToolInputRejected, ToolRegistry
from src.scoring import shap_explainer as S


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionIn(_Model):
    match: MatchResult


class PredictionOut(_Model):
    student_id: str
    job_id: str
    job_title: str | None
    suitability: float
    features: dict[str, float | None]
    semantic_raw: float | None
    model_versions: PredictionVersions


class ShapIn(_Model):
    features: dict[str, float | None]
    model_versions: PredictionVersions
    suitability: float


class ShapOut(_Model):
    matching_model: str
    matching_model_version: str
    explainer: str
    output_space: str
    base_value: float
    model_output: float
    suitability: float
    values: dict[str, float]
    additivity_error: float


class Contribution(_Model):
    feature: str
    shap: float
    share: float  # |shap| / sum of |shap| over all features for this pair
    tied_with: list[str]


class ContributionsIn(_Model):
    values: dict[str, float]
    output_space: str
    matching_model: str


class ContributionsOut(_Model):
    positive: list[Contribution]
    negative: list[Contribution]
    omitted: int  # features below min_abs_share or beyond max_per_direction
    total_abs: float
    min_abs_share: float
    max_per_direction: int
    output_space: str
    values: dict[str, float]
    skill_feature_caveat: str | None


class EvidenceIn(_Model):
    prediction: PredictionOut
    skill_gap_report: SkillGapReport
    contributions: ContributionsOut


class GenerateIn(_Model):
    evidence: EvidenceBundle
    strict: bool = False
    previous_errors: list[str] = []


class GenerateOut(_Model):
    draft: ProseDraft | None
    llm: str | None
    reason: str | None  # why there is no draft


class ValidateIn(_Model):
    evidence: EvidenceBundle
    draft: ProseDraft
    attempt: int


class MatchExplanationIn(_Model):
    prediction: PredictionOut
    shap: ShapOut
    evidence: EvidenceBundle
    draft: ProseDraft | None
    validations: list[ExplanationValidation]
    llm: str | None
    llm_attempts: int = 0  # generate_explanation calls that reached the LLM
    fallback_reason: str | None


class GlobalIn(_Model):
    embedding_model: str | None = None


def register_explainability_tools(registry: ToolRegistry, *, repo, settings: Settings, llm: LLMClient | None,
                                  default_embedding_model: str,
                                  model_loader: Callable[[str, str | None], Any] = S.load_model) -> None:
    llm = traced(llm)  # each explanation call is a trace span inside its tool attempt (Phase 12)
    models: dict[tuple[str, str | None], Any] = {}

    def model_for(embedding_model: str, tool: str):
        key = (embedding_model, settings.matching_model)
        if key not in models:
            try:
                models[key] = model_loader(embedding_model, settings.matching_model)
            except ValueError as exc:  # no model configured for these embeddings, missing or tampered artifact
                raise ToolInputRejected(tool, str(exc)) from exc
        return models[key]

    @registry.register("get_model_prediction", input_schema=PredictionIn, output_schema=PredictionOut)
    def tool_get_model_prediction(match):
        """The prediction to explain, with its feature vector and version metadata; refuses an incomplete one."""
        if not isinstance(match["suitability_score"], float | int):
            raise ToolInputRejected("get_model_prediction", "the match has no suitability score to explain")
        if isinstance(match["trace"], str):
            raise ToolInputRejected("get_model_prediction", "the match carries no feature trace")
        trace = MatchTrace.model_validate(match["trace"])
        if set(trace.features) != set(FEATURE_NAMES):
            raise ToolInputRejected("get_model_prediction", "the trace is not a complete features-v1 vector")
        if isinstance(match["skill_gap"], str):
            raise ToolInputRejected("get_model_prediction", "the match carries no skill-gap report")
        return {"student_id": match["student_id"], "job_id": match["job_id"],
                "job_title": (match.get("summary") or {}).get("job_title"),
                "suitability": float(match["suitability_score"]), "features": trace.features,
                "semantic_raw": trace.retrieval.get("raw_score"), "model_versions": match["model_versions"]}

    @registry.register("get_shap_values", input_schema=ShapIn, output_schema=ShapOut)
    def tool_get_shap_values(features, model_versions, suitability):
        """SHAP values for this feature vector under the model that made the prediction (deterministic)."""
        model = model_for(model_versions["embedding_model"], "get_shap_values")
        if model.version != model_versions["matching_model_version"]:
            raise ToolInputRejected("get_shap_values", f"the prediction came from "
                                    f"{model_versions['matching_model_version']}, the loaded model is {model.version}")
        try:
            out = S.explain_pair(model, features)
        except S.ExplanationUnavailable as exc:
            raise ToolInputRejected("get_shap_values", str(exc)) from exc
        if abs(out["suitability"] - suitability) > 1e-6:
            raise ToolInputRejected("get_shap_values", f"the model scores this vector {out['suitability']}, the "
                                                       f"prediction says {suitability}: not the same prediction")
        return out

    @registry.register("get_feature_contributions", input_schema=ContributionsIn, output_schema=ContributionsOut)
    def tool_get_feature_contributions(values, output_space, matching_model):
        """The pair's largest contributions per direction (share of total |SHAP| >= min_abs_share)."""
        cfg = S.explainability_config()["individual"]
        min_share, top = float(cfg["min_abs_share"]), int(cfg["max_per_direction"])
        total = sum(abs(v) for v in values.values())
        ranked = sorted(values.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
        chosen = {1: [], -1: []}
        for feature, v in ranked:
            share = abs(v) / total if total else 0.0
            sign = 1 if v > 0 else -1
            if v != 0 and share >= min_share and len(chosen[sign]) < top:
                chosen[sign].append({"feature": feature, "shap": round(v, 6), "share": round(share, 4)})
        for side in chosen.values():
            for c in side:
                c["tied_with"] = sorted(f for f, v in values.items() if f != c["feature"] and round(v, 6) == c["shap"])
        picked = {c["feature"] for side in chosen.values() for c in side}
        caveat = None
        if matching_model != S.BASELINE and picked & set(S.SKILL_FEATURES):
            caveat = ("Skill features (counts and ratios in features-v1) have small, sometimes negative contributions "
                      "under this tree ensemble, whose output moves in steps (docs/adr/0015, 0016): approximate.")
        return {"positive": chosen[1], "negative": chosen[-1], "omitted": len(values) - len(picked),
                "total_abs": round(total, 6), "min_abs_share": min_share, "max_per_direction": top,
                "output_space": output_space, "values": values, "skill_feature_caveat": caveat}

    @registry.register("generate_evidence", input_schema=EvidenceIn, output_schema=EvidenceBundle)
    def tool_generate_evidence(prediction, skill_gap_report, contributions):
        """One evidence item per claim, from the profile, the posting, the skill-gap report and the SHAP values."""
        student = job = None
        if repo is not None:
            try:
                found = repo.student(prediction["student_id"])
                student = found[0] if found else None  # [1] is the visibility gender: never used here
                job = repo.job(prediction["job_id"])
            except Exception:  # noqa: BLE001 - records only add detail; the features carry the comparison
                student = job = None
        cfg = S.explainability_config()["individual"]
        return build_evidence(
            suitability=prediction["suitability"], features=prediction["features"],
            report=SkillGapReport.model_validate(skill_gap_report), contributions=contributions,
            semantic_raw=prediction["semantic_raw"], student=student, job=job,
            positive_similarity_percentile=float(cfg["positive_similarity_percentile"]))

    @registry.register("generate_explanation", input_schema=GenerateIn, output_schema=GenerateOut)
    async def tool_generate_explanation(evidence, strict=False, previous_errors=()):
        """The LLM turns the evidence into cited lines (it computes nothing). No LLM or a failed call: no draft."""
        if llm is None:
            return {"draft": None, "llm": None, "reason": "no LLM configured (GEMINI_API_KEY, GEMINI_MODEL)"}
        bundle = EvidenceBundle.model_validate(evidence)
        cfg = S.explainability_config()["individual"]
        try:
            draft = await llm.generate_structured(
                system=prose.system_prompt(), prompt=prose.build_prompt(bundle, strict, list(previous_errors)),
                schema=ProseDraft, timeout_s=max(1.0, min(settings.tool_timeout_seconds - 2, 25.0)),
                max_output_tokens=int(cfg["max_output_tokens"]))
        except LLMError as exc:  # short: logged agent errors longer than 120 characters are redacted
            first = str(exc).split(". {", 1)[0].split("\n", 1)[0]  # "ClientError: 429 RESOURCE_EXHAUSTED"
            return {"draft": None, "llm": llm.model_name, "reason": f"{type(exc).__name__}: {first}"[:70]}
        return {"draft": draft.model_dump(), "llm": llm.model_name, "reason": None}

    @registry.register("validate_explanation", input_schema=ValidateIn, output_schema=ExplanationValidation)
    def tool_validate_explanation(evidence, draft, attempt):
        """Check every line against the evidence it cites; any unbacked claim fails the whole draft."""
        return validate(EvidenceBundle.model_validate(evidence), ProseDraft.model_validate(draft), attempt)

    @registry.register("get_match_explanation", input_schema=MatchExplanationIn, output_schema=MatchExplanation)
    def tool_get_match_explanation(prediction, shap, evidence, draft, validations, llm, llm_attempts, fallback_reason):
        """The explanation in the spec's format: the validated draft, else the structured evidence itself."""
        bundle = EvidenceBundle.model_validate(evidence)
        if draft is not None:
            check = validate(bundle, ProseDraft.model_validate(draft), attempt=len(validations))
            if not check.valid:  # re-checked here: an unvalidated sentence is never returned
                raise ToolInputRejected("get_match_explanation", "the draft did not pass validate_explanation")
            final, source = ProseDraft.model_validate(draft), "llm_validated"
        else:
            final, source = prose.structured_draft(bundle), "structured_fallback"
            check = validate(bundle, final, attempt=0)
            if not check.valid:
                raise RuntimeError(f"the structured evidence failed its own validation: {check.errors[:3]}")
            validations = [*validations, check.model_dump()]
        lines = prose.rendered_lines(bundle, final)
        contributions = {k: shap[k] for k in ("explainer", "output_space", "base_value", "model_output", "values",
                                              "additivity_error")}
        lines["model_evidence"].append(prose.output_line(contributions))  # deterministic, after validation
        return {
            "student_id": prediction["student_id"], "job_id": prediction["job_id"],
            "job_title": prediction["job_title"], "suitability_score": prediction["suitability"], **lines,
            "text": prose.render(prediction["suitability"], lines), "source": source,
            "generation": {"llm": llm, "llm_attempts": llm_attempts,
                           "validations": validations, "fallback_reason": fallback_reason if draft is None else None,
                           "suitability_note": "the model's score for this pair; class-balanced training makes it a "
                                               "ranking score, not a calibrated probability (docs/adr/0015)"},
            "evidence": bundle.model_dump(),
            "contributions": contributions,
            "model_versions": prediction["model_versions"],
        }

    @registry.register("get_global_explanation", input_schema=GlobalIn, output_schema=GlobalExplanation)
    def tool_get_global_explanation(embedding_model=None):
        """The active model as a whole: importance ranking, SHAP distributions, summary-plot data."""
        model = model_for(embedding_model or default_embedding_model, "get_global_explanation")
        try:
            return _global(model)
        except S.ExplanationUnavailable as exc:
            raise ToolInputRejected("get_global_explanation", str(exc)) from exc

    cache: dict[str, dict] = {}

    def _global(model) -> dict:  # one computation per model version per process
        if model.version not in cache:
            cache[model.version] = S.global_explanation(model)
        return cache[model.version]
