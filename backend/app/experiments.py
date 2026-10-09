"""Read-only access to recorded experiments (spec §25, experiment_runs / experiment_metrics).

Experiments are written by the evaluation engines (Phase 6: src/embeddings/model_evaluation_engine.py),
never by the API.

    GET /experiments?kind=embedding_selection   newest first, summary fields only
    GET /experiments/{experiment_id}            the full record, including the engine's report
"""

from typing import Any

import psycopg
from fastapi import APIRouter, HTTPException
from psycopg.rows import dict_row

router = APIRouter(prefix="/experiments", tags=["experiments"])
_dsn: str | None = None


def configure(dsn: str | None) -> None:
    global _dsn
    _dsn = dsn


def _query(sql: str, params: tuple) -> list[dict[str, Any]]:
    if not _dsn:
        raise HTTPException(503, "experiments need the database (DATABASE_URL is not set)")
    try:
        with psycopg.connect(_dsn, connect_timeout=5, row_factory=dict_row) as conn:
            return conn.execute(sql, params).fetchall()
    except psycopg.Error:
        raise HTTPException(503, "database unavailable") from None


@router.get("")
def list_experiments(kind: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    rows = _query(
        "SELECT experiment_id, kind, finished_at, dataset_version, labels, "
        "report -> 'best_models' AS best_models, report -> 'validity' AS validity "
        "FROM experiment_runs WHERE (%s::text IS NULL OR kind = %s) ORDER BY finished_at DESC LIMIT %s",
        (kind, kind, max(1, min(limit, 100))))
    return rows


@router.get("/{experiment_id}")
def get_experiment(experiment_id: str) -> dict[str, Any]:
    rows = _query("SELECT * FROM experiment_runs WHERE experiment_id = %s", (experiment_id,))
    if not rows:
        raise HTTPException(404, "unknown experiment_id")
    record = rows[0]
    record["metric_rows"] = _query("SELECT model_name, model_version, metric, split, value, status "
                                   "FROM experiment_metrics WHERE experiment_id = %s ORDER BY model_name, metric",
                                   (experiment_id,))
    return record
