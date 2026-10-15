"""Admin / Research dashboard API (Phase 13, spec §30 + §31). Admin role only.

Nothing here is hardcoded: the leaderboards, metric charts and experiment history read experiment_runs and
experiment_metrics (written by the Phase 6 and Phase 8 engines with --record-db / --record-existing), so a re-run
with new labels shows up on its own. SHAP comes from the Explainability Agent's global pipeline (Phase 11), traces
from agent_runs + agent_messages (Phase 12).

    GET /admin/overview                     deployed models, dataset/model versions, benchmark configuration, counts
    GET /admin/leaderboard/{kind}           spec §31 columns for one experiment (default: newest human-label run)
    GET /admin/metrics/{kind}               every recorded metric row of every run of that kind (charts, history)
    GET /admin/experiments                  experiment history (both engines)
    GET /admin/experiments/{id}             one experiment, including its report
    GET /admin/shap                         get_global_explanation output for the active matching model
    GET /admin/traces                       recent Supervisor requests (filter by task, status)
    GET /admin/traces/{request_id}          get_agent_trace(request_id)
    GET /admin/agent-latency                per agent and per tool latency, from the persisted traces
"""

import uuid
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import yaml
from fastapi import APIRouter, Depends, HTTPException

from app.auth import require_role
from app.core.trace_tree import get_agent_trace
from app.portal.service import Portal
from app.schemas.agent_io import ExplainabilityAgentInput
from app.schemas.agent_state import AgentState

router = APIRouter(prefix="/admin", tags=["admin / research dashboard"])
_portal: Portal | None = None
_embedder = None
Admin = Depends(require_role("admin"))
CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

Kind = Literal["embedding", "matching"]
KIND_DB = {"embedding": "embedding_selection", "matching": "matching_selection"}

# spec §31, column for column. Each maps to the metric the engine recorded; "quality" for embeddings is the engine's
# own primary quality metric (config.evaluation.primary_k, gate.primary_quality_metric = ndcg).
COLUMNS: dict[str, list[dict[str, Any]]] = {
    "embedding": [
        {"key": "quality", "label": "Quality", "metric": "ndcg@{primary_k}", "fmt": "score",
         "note": "the engine's primary quality metric (NDCG@{primary_k}, validation split)"},
        {"key": "mrr", "label": "MRR", "metric": "mrr", "fmt": "score"},
        {"key": "ndcg", "label": "NDCG@K", "metric": "ndcg@{k}", "fmt": "score", "per_k": True},
        {"key": "recall", "label": "Recall@K", "metric": "recall@{k}", "fmt": "score", "per_k": True},
        {"key": "precision", "label": "Precision@K", "metric": "precision@{k}", "fmt": "score", "per_k": True},
        {"key": "latency", "label": "Latency", "metric": "latency", "fmt": "ms", "unit": "ms / student-job pair (CPU, median)"},
        {"key": "memory", "label": "Memory", "metric": "memory", "fmt": "mb", "unit": "MB peak added"},
        {"key": "size", "label": "Model Size", "metric": "size", "fmt": "mb", "unit": "MB on disk"},
        {"key": "dimension", "label": "Dimension", "metric": "dimension", "fmt": "int",
         "unit": "stored dimensions per pair, as the engine scored it"},
        {"key": "overall", "label": "Overall Score", "metric": "overall_score_balanced", "fmt": "score",
         "unit": "balanced weighting scenario"},
    ],
    "matching": [
        {"key": "accuracy", "label": "Accuracy", "metric": "accuracy", "fmt": "score"},
        {"key": "precision", "label": "Precision", "metric": "precision", "fmt": "score"},
        {"key": "recall", "label": "Recall", "metric": "recall", "fmt": "score"},
        {"key": "f1", "label": "F1", "metric": "f1", "fmt": "score"},
        {"key": "ndcg", "label": "NDCG@K", "metric": "ndcg@{k}", "fmt": "score", "per_k": True},
        {"key": "mrr", "label": "MRR", "metric": "mrr", "fmt": "score"},
        {"key": "latency", "label": "Latency", "metric": "latency", "fmt": "ms", "unit": "ms / pair (inference)"},
        {"key": "training_time", "label": "Training Time", "metric": "training_time", "fmt": "s", "unit": "s (median of refits)"},
        {"key": "memory", "label": "Memory", "metric": "memory", "fmt": "mb", "unit": "MB added in training"},
        {"key": "explainability", "label": "Explainability", "metric": "explainability", "fmt": "int",
         "unit": "points (engine rubric)"},
        {"key": "overall", "label": "Overall Score", "metric": "overall_score_balanced", "fmt": "score",
         "unit": "balanced weighting scenario"},
    ],
}


def configure(portal: Portal, embedder) -> None:
    global _portal, _embedder
    _portal, _embedder = portal, embedder


def portal() -> Portal:
    if _portal is None:
        raise HTTPException(503, "portal not configured")
    return _portal


def _iso(v):
    return v.isoformat() if hasattr(v, "isoformat") else v


def _experiment(kind: Kind, experiment_id: str | None) -> dict[str, Any]:
    p = portal()
    if experiment_id:
        row = p.one("SELECT * FROM experiment_runs WHERE experiment_id = %s AND kind = %s", (experiment_id, KIND_DB[kind]))
        if row is None:
            raise HTTPException(404, "unknown experiment for this kind")
        return row
    # Default: the newest run evaluated on human labels (the one a selection is made on), else the newest run.
    row = p.one("SELECT * FROM experiment_runs WHERE kind = %s AND labels->>'used_source' = 'human_reviewed' "
                "ORDER BY finished_at DESC LIMIT 1", (KIND_DB[kind],))
    row = row or p.one("SELECT * FROM experiment_runs WHERE kind = %s ORDER BY finished_at DESC LIMIT 1", (KIND_DB[kind],))
    if row is None:
        raise HTTPException(404, f"No {kind} experiment recorded yet. Run the engine with --record-db.")
    return row


def _summary(row: dict[str, Any]) -> dict[str, Any]:
    report = row.get("report") or {}
    labels = row.get("labels") or {}
    return {
        "experiment_id": row["experiment_id"], "kind": row["kind"], "finished_at": _iso(row["finished_at"]),
        "started_at": _iso(row["started_at"]), "runtime_seconds": row["runtime_seconds"],
        "dataset_version": row["dataset_version"], "dataset_size": row["dataset_size"],
        "feature_version": row["feature_version"], "embedding_model": row["embedding_model"],
        "embedding_version": row["embedding_version"], "prediction_model": row["prediction_model"],
        "random_seed": row["random_seed"], "hardware": row["hardware"],
        "labels": {k: labels.get(k) for k in ("provided", "path", "sha256", "pairs", "used_source", "used_split",
                                               "used_pairs", "used_students", "stamp")},
        "validity": report.get("validity") or (labels.get("validity") or {}).get("line"),
        "recommendation": report.get("recommendation"), "best_models": report.get("best_models"),
        "config": report.get("config"), "notes": row["notes"], "limitations": row["limitations"],
    }


@router.get("/leaderboard/{kind}")
def leaderboard(kind: Kind, experiment_id: str | None = None, _: dict = Admin):
    row = _experiment(kind, experiment_id)
    report = row.get("report") or {}
    evaluation = ((report.get("config") or {}).get("evaluation") or {})
    ks = evaluation.get("ks") or [5, 10]
    primary_k = evaluation.get("primary_k") or ks[0]
    metrics = portal().query("SELECT model_name, model_version, metric, split, value, status FROM experiment_metrics "
                             "WHERE experiment_id = %s", (row["experiment_id"],))
    by_model: dict[str, dict[str, dict]] = {}
    versions: dict[str, str] = {}
    for m in metrics:
        by_model.setdefault(m["model_name"], {})[m["metric"]] = {"value": m["value"], "status": m["status"],
                                                                 "split": m["split"]}
        versions[m["model_name"]] = m["model_version"]
    models_meta = report.get("models") or {}
    order = list(models_meta) + sorted(set(by_model) - set(models_meta))
    columns = []
    for c in COLUMNS[kind]:
        c = dict(c)
        c["metric"] = c["metric"].replace("{primary_k}", str(primary_k))
        if "note" in c:
            c["note"] = c["note"].replace("{primary_k}", str(primary_k))
        columns.append(c)
    recommendation = report.get("recommendation") or {}
    eligible = set(report.get("eligible") or [])
    rows = []
    for model in order:
        meta = models_meta.get(model) or {}
        got = by_model.get(model, {})
        cells: dict[str, Any] = {}
        for c in columns:
            names = [c["metric"].replace("{k}", str(k)) for k in ks] if c.get("per_k") else [c["metric"]]
            cells[c["key"]] = {n: got.get(n, {"value": None, "status": "not_yet_evaluated", "split": None})
                               for n in names}
        rows.append({"model": model, "display": meta.get("display") or model, "role": meta.get("role"),
                     "slot": meta.get("slot"), "version": versions.get(model) or meta.get("version"),
                     "selected": recommendation.get("model") == model and recommendation.get("decision") == "select",
                     "eligible": (model in eligible) if eligible else None, "cells": cells})
    not_run = report.get("not_run") or {}
    return {"experiment": _summary(row), "ks": ks, "primary_k": primary_k, "columns": columns, "rows": rows,
            "not_run": not_run}


@router.get("/metrics/{kind}")
def metrics(kind: Kind, _: dict = Admin):
    rows = portal().query(
        "SELECT r.experiment_id, r.finished_at, r.labels->>'used_source' AS label_source, m.model_name, m.metric, "
        "m.split, m.value, m.status FROM experiment_metrics m JOIN experiment_runs r USING (experiment_id) "
        "WHERE r.kind = %s ORDER BY r.finished_at, m.model_name, m.metric", (KIND_DB[kind],))
    return [{**r, "finished_at": _iso(r["finished_at"])} for r in rows]


@router.get("/experiments")
def experiments(_: dict = Admin):
    rows = portal().query("SELECT * FROM experiment_runs ORDER BY finished_at DESC")
    counts = {r["experiment_id"]: r["n"] for r in portal().query(
        "SELECT experiment_id, count(*) AS n FROM experiment_metrics GROUP BY experiment_id")}
    return [{**_summary(r), "metric_rows": counts.get(r["experiment_id"], 0)} for r in rows]


@router.get("/experiments/{experiment_id}")
def experiment(experiment_id: str, _: dict = Admin):
    row = portal().one("SELECT * FROM experiment_runs WHERE experiment_id = %s", (experiment_id,))
    if row is None:
        raise HTTPException(404, "unknown experiment_id")
    metric_rows = portal().query("SELECT model_name, model_version, metric, split, value, status FROM "
                                 "experiment_metrics WHERE experiment_id = %s ORDER BY model_name, metric",
                                 (experiment_id,))
    return {**_summary(row), "metric_rows": metric_rows}


def _yaml(name: str) -> dict:
    try:
        return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8")) or {}
    except OSError:
        return {}


@router.get("/overview")
def overview(_: dict = Admin):
    p = portal()
    counts = p.one(
        "SELECT (SELECT count(*) FROM student_profiles WHERE profile IS NOT NULL) AS profiles, "
        "(SELECT count(*) FROM student_profiles) AS students, (SELECT count(*) FROM jobs) AS jobs, "
        "(SELECT count(*) FROM jobs WHERE posted_by IS NOT NULL) AS portal_jobs, "
        "(SELECT count(*) FROM users) AS users, (SELECT count(*) FROM experiment_runs) AS experiments, "
        "(SELECT count(*) FROM agent_runs WHERE agent_name = 'supervisor') AS requests, "
        "(SELECT count(*) FROM agent_messages) AS trace_messages, (SELECT count(*) FROM feedback_events) AS feedback, "
        "(SELECT count(*) FROM relevance_judgments) AS judgments, (SELECT count(*) FROM learning_resources) AS resources")
    embeddings = p.query("SELECT entity_type, model_name, model_version, count(DISTINCT entity_id) AS entities "
                         "FROM embeddings GROUP BY 1, 2, 3 ORDER BY 1, 2")
    latest = {}
    for kind in KIND_DB:
        try:
            latest[kind] = _summary(_experiment(kind, None))
        except HTTPException:
            latest[kind] = None
    version, dimension = (_embedder.describe() if _embedder is not None else (None, None))
    return {
        "counts": counts, "embeddings": embeddings,
        "deployed": {
            "embedding_model": getattr(_embedder, "model_name", None), "embedding_version": version,
            "embedding_dimension": dimension, "embedding_fallback": getattr(_embedder, "fallback_reason", None),
            "matching": _yaml("matching_model.yaml"),
        },
        "latest_experiments": latest,
        "benchmark_configuration": {
            "embedding_selection_weights": _yaml("model_selection_weights.yaml"),
            "matching_selection_weights": _yaml("matching_model_selection_weights.yaml"),
            "matching_candidates": _yaml("matching_model_candidates.yaml"),
        },
    }


@router.get("/shap")
async def shap(_: dict = Admin):
    """The Explainability Agent's global pipeline (get_global_explanation), run through the agent itself."""
    agent = portal().quiet.agents["explainability_agent"]
    state = AgentState(request_id=str(uuid.uuid4()), task_type="explain_global", status="running")
    out = await agent.run(state, ExplainabilityAgentInput(mode="global"))
    data = out.data if isinstance(out.data, dict) else None
    g = (data or {}).get("global_explanation")
    if not isinstance(g, dict):
        raise HTTPException(503, f"global explanation unavailable ({out.termination_reason}): "
                                 + "; ".join(e.message for e in out.errors))
    return {"global_explanation": g, "duration_ms": out.duration_ms, "status": out.status}


@router.get("/traces")
def traces(task: str | None = None, status: str | None = None, limit: int = 50, _: dict = Admin):
    rows = portal().query(
        "SELECT r.request_id, r.task_type, r.status, r.started_at, r.duration_ms, r.routing_source, r.model_used, "
        "r.student_id, r.job_id, r.agents_called, r.retry_count, "
        "(SELECT count(*) FROM agent_runs c WHERE c.parent_run_id = r.request_id) AS executions "
        "FROM agent_runs r WHERE r.agent_name = 'supervisor' AND r.parent_run_id IS NULL "
        "AND (%(task)s::text IS NULL OR r.task_type = %(task)s) AND (%(status)s::text IS NULL OR r.status = %(status)s) "
        "ORDER BY r.started_at DESC LIMIT %(limit)s",
        {"task": task, "status": status, "limit": max(1, min(limit, 200))})
    return [{**r, "request_id": str(r["request_id"]), "started_at": _iso(r["started_at"])} for r in rows]


@router.get("/traces/{request_id}")
async def trace(request_id: UUID, _: dict = Admin):
    store = portal().quiet.store
    if store is None:
        raise HTTPException(503, "persistence is not configured")
    tree = await get_agent_trace(str(request_id), store)
    if tree is None:
        raise HTTPException(404, "unknown request_id")
    return tree.model_dump(mode="json")


@router.get("/agent-latency")
def agent_latency(_: dict = Admin):
    agents = portal().query(
        "SELECT agent_name, count(*) AS runs, avg(duration_ms) AS mean_ms, "
        "percentile_cont(0.5) WITHIN GROUP (ORDER BY duration_ms) AS p50_ms, "
        "percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms, "
        "sum(CASE WHEN status <> 'success' THEN 1 ELSE 0 END) AS not_success "
        "FROM agent_runs GROUP BY agent_name ORDER BY mean_ms DESC")
    tools = portal().query(
        "SELECT recipient AS name, message_type, count(*) AS calls, avg(duration_ms) AS mean_ms, "
        "percentile_cont(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_ms, "
        "sum(CASE WHEN status NOT IN ('success', 'ok') THEN 1 ELSE 0 END) AS not_success "
        "FROM agent_messages WHERE message_type IN ('tool_call', 'llm_call') GROUP BY 1, 2 ORDER BY mean_ms DESC")
    return {"agents": agents, "tools": tools}
