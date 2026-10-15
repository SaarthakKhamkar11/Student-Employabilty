"""The dashboards' access to the agents (Phase 13).

Every dashboard value comes from a real Supervisor run (Phases 3-12), never from a separate code path:

    Portal.result(task, student_id, job_id)
      ├─ a stored run: the newest successful agent_runs row for the same task and ids, finished after the last
      │   change to any profile, posting or confirmation (so new data invalidates it), or
      └─ a fresh run: SupervisorAgent.handle(...) with the explicit task_type, persisted with its trace as usual

Two Supervisors share the process's embedding models:
- `quiet`: built without an LLM. Dashboards load through it, so a page view never spends Gemini quota: routing is
  deterministic (explicit task types) and the Explainability Agent returns its validated structured evidence.
- `llm`: the configured Gemini chain, used only by the explicit "Ask AI" action (cached in llm_explanations).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import psycopg
from fastapi import HTTPException
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.agents import SupervisorAgent
from app.core.logging import get_logger, log_event
from app.schemas.agent_io import SupervisorInput, SupervisorResponse

logger = get_logger("portal")

QUOTA_MESSAGE = "AI summary unavailable (quota exceeded). Showing stored explanation."


class Portal:
    def __init__(self, quiet: SupervisorAgent, llm: SupervisorAgent, dsn: str | None, llm_configured: bool):
        self.quiet = quiet
        self.llm = llm
        self.llm_configured = llm_configured
        self.dsn = dsn
        self._locks: dict[tuple, asyncio.Lock] = {}

    # -- database ----------------------------------------------------------------------------------------------
    def connect(self) -> psycopg.Connection:
        if not self.dsn:
            raise HTTPException(503, "the portal needs the database (DATABASE_URL is not set)")
        try:
            return psycopg.connect(self.dsn, connect_timeout=5, row_factory=dict_row)
        except psycopg.Error:
            raise HTTPException(503, "database unavailable") from None

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple | dict = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def audit(self, user_id, action: str, entity_type: str, entity_id: str, details: dict, request_id=None) -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO audit_logs (actor_type, actor_user_id, action, entity_type, entity_id, "
                         "request_id, details) VALUES ('user', %s, %s, %s, %s, %s, %s)",
                         (user_id, action, entity_type, entity_id, request_id, Jsonb(details)))

    # -- runs --------------------------------------------------------------------------------------------------
    def _matching_model_version(self) -> str | None:
        """The active matching model's version (config/matching_model.yaml, read once per process: the model only
        changes with a re-selection and a restart). None if it can't be built; then only data changes count."""
        if not hasattr(self, "_model_version"):
            try:
                from app.core.config import get_settings
                from app.matching.config import get_matching_config
                from app.matching.predictor import build_predictor

                predictor = build_predictor(get_matching_config(), embedding_model=get_settings().embedding_model)
                self._model_version = predictor.version
            except Exception:  # noqa: BLE001 - staleness check only; never block a dashboard on it
                self._model_version = None
        return self._model_version

    def _data_changed_at(self) -> datetime | None:
        """The last change to anything a stored result depends on: any profile, posting or confirmation."""
        row = self.one(
            "SELECT greatest((SELECT max(profile_updated_at) FROM student_profiles), "
            "(SELECT max(posting_updated_at) FROM jobs), (SELECT max(confirmed_at) FROM jobs), "
            "(SELECT max(profile_confirmed_at) FROM users)) AS t")
        return row["t"] if row else None

    def stored(self, tasks: tuple[str, ...], student_id: str | None, job_id: str | None,
               current_only: bool = True) -> dict[str, Any] | None:
        """The newest stored run; with current_only=False, also one that newer data has since made out of date
        (summaries that show its timestamp)."""
        changed = self._data_changed_at() if current_only else None
        # a run scored by another matching model (before a re-selection) is out of date too
        model = self._matching_model_version() if current_only else None
        row = self.one(
            "SELECT request_id, task_type, status, finished_at, duration_ms, agent_results, model_versions, errors, "
            "routing_source, model_used FROM agent_runs WHERE agent_name = 'supervisor' AND parent_run_id IS NULL "
            "AND task_type = ANY(%s) AND student_id IS NOT DISTINCT FROM %s AND job_id IS NOT DISTINCT FROM %s "
            "AND status IN ('success', 'partial') AND (%s::timestamptz IS NULL OR finished_at > %s::timestamptz) "
            "AND (%s::text IS NULL OR model_versions->>'matching_model_version' IS NULL "
            "     OR model_versions->>'matching_model_version' = %s::text) "
            "ORDER BY finished_at DESC LIMIT 1",
            (list(tasks), student_id, job_id, changed, changed, model, model))
        return _from_row(row) if row else None

    async def result(self, task: str, *, actor: str, student_id: str | None = None, job_id: str | None = None,
                     refresh: bool = False, also_accept: tuple[str, ...] = (), user_id=None) -> dict[str, Any]:
        """The stored result of `task` for these ids, or a fresh Supervisor run when none is current."""
        key = (task, student_id, job_id)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:  # two tabs asking at once share one run
            if not refresh:
                found = await asyncio.to_thread(self.stored, (task, *also_accept), student_id, job_id)
                if found is not None:
                    return found
            request = SupervisorInput(actor=actor, request_kind="question", task_type=task,
                                      request_text=f"dashboard: {task}", student_id=student_id, job_id=job_id)
            response = await self.quiet.handle(request, user_id=str(user_id) if user_id else None)
            return from_response(response)

    async def run(self, request: SupervisorInput, user_id=None, llm: bool = False) -> dict[str, Any]:
        sup = self.llm if llm else self.quiet
        return from_response(await sup.handle(request, user_id=str(user_id) if user_id else None))

    # -- Ask AI ------------------------------------------------------------------------------------------------
    def cached_ai(self, student_id: str, job_id: str, versions: dict | None) -> dict[str, Any] | None:
        if not versions:
            return None
        row = self.one(
            "SELECT explanation, llm_model, request_id, created_at FROM llm_explanations WHERE student_id = %s "
            "AND job_id = %s AND matching_model_version = %s AND embedding_version = %s",
            (student_id, job_id, versions.get("matching_model_version"), versions.get("embedding_version")))
        if row is None:
            return None
        return {"explanation": row["explanation"], "llm_model": row["llm_model"], "request_id": str(row["request_id"]),
                "created_at": row["created_at"].isoformat(), "cached": True}

    async def ask_ai(self, *, actor: str, student_id: str, job_id: str, user_id) -> dict[str, Any]:
        """Gemini prose for one prediction, on request only. Cached on success; never fabricated on failure."""
        pair = await self.result("explain_match", actor=actor, student_id=student_id, job_id=job_id, user_id=user_id)
        versions = _pair_versions(pair)
        cached = await asyncio.to_thread(self.cached_ai, student_id, job_id, versions)
        if cached is not None:
            return {"status": "ok", **cached}
        if not self.llm_configured:
            return {"status": "unavailable", "message": "AI summary unavailable (no LLM configured). "
                    "Showing stored explanation.", "reason": "no LLM configured"}
        request = SupervisorInput(actor=actor, request_kind="question", task_type="explain_match",
                                  request_text="Explain this match", student_id=student_id, job_id=job_id)
        run = await self.run(request, user_id=user_id, llm=True)
        explanation = _agent_data(run, "explainability_agent", "match_explanation")
        if not isinstance(explanation, dict):
            return {"status": "unavailable", "message": "AI summary unavailable (the explanation could not be "
                    "produced). Showing stored explanation.", "reason": run.get("summary"),
                    "request_id": run["request_id"]}
        generation = explanation.get("generation") or {}
        if explanation.get("source") != "llm_validated":
            reason = str(generation.get("fallback_reason") or "the LLM returned no usable draft")
            quota = "429" in reason or "quota" in reason.lower() or "RESOURCE_EXHAUSTED" in reason
            message = QUOTA_MESSAGE if quota else f"AI summary unavailable ({reason}). Showing stored explanation."
            log_event(logger, "ask_ai_unavailable", logging.WARNING, reason=reason[:120])
            return {"status": "unavailable", "message": message, "reason": reason, "request_id": run["request_id"]}
        mv = explanation.get("model_versions") or {}
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO llm_explanations (student_id, job_id, matching_model_version, embedding_version, "
                "llm_model, request_id, explanation) VALUES (%s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (student_id, job_id, matching_model_version, embedding_version) DO UPDATE SET "
                "explanation = EXCLUDED.explanation, llm_model = EXCLUDED.llm_model, "
                "request_id = EXCLUDED.request_id, created_at = now()",
                (student_id, job_id, mv.get("matching_model_version"), mv.get("embedding_version"),
                 generation.get("llm") or "gemini", run["request_id"], Jsonb(explanation)))
        return {"status": "ok", "explanation": explanation, "llm_model": generation.get("llm"),
                "request_id": run["request_id"], "cached": False}


# -- normalised run shape ---------------------------------------------------------------------------------------
def from_response(r: SupervisorResponse) -> dict[str, Any]:
    agents = {name: {"status": o.status, "data": o.data, "skip_reason": o.skip_reason, "error_codes": o.error_codes}
              for name, o in r.results.items()}
    return {"request_id": r.request_id, "status": r.status, "task_type": r.task_type, "stored": False,
            "finished_at": r.finished_at.isoformat(), "duration_ms": r.duration_ms, "summary": r.summary,
            "model_versions": r.model_versions, "errors": r.errors, "agents": agents,
            "routing": r.routing if isinstance(r.routing, str) else r.routing.model_dump(mode="json")}


def _from_row(row: dict[str, Any]) -> dict[str, Any]:
    agents = {}
    for name, out in (row["agent_results"] or {}).items():
        if name == "supervisor" or not isinstance(out, dict):
            continue
        agents[name] = {"status": out.get("status"), "data": out.get("data"), "skip_reason": None,
                        "error_codes": [e.get("code") for e in out.get("errors") or [] if isinstance(e, dict)]}
    return {"request_id": str(row["request_id"]), "status": row["status"], "task_type": row["task_type"],
            "stored": True, "finished_at": row["finished_at"].isoformat(), "duration_ms": row["duration_ms"],
            "summary": None, "model_versions": row["model_versions"], "errors": row["errors"], "agents": agents,
            "routing": {"source": row["routing_source"], "model_used": row["model_used"]}}


def _agent_data(run: dict[str, Any], agent: str, field: str | None = None) -> Any:
    a = run.get("agents", {}).get(agent)
    if not a or not isinstance(a.get("data"), dict):
        return None
    return a["data"].get(field) if field else a["data"]


def agent_data(run: dict[str, Any], agent: str, field: str | None = None) -> Any:
    return _agent_data(run, agent, field)


def _pair_versions(run: dict[str, Any]) -> dict[str, Any] | None:
    matches = _agent_data(run, "matching_agent", "matches")
    if isinstance(matches, list) and matches:
        return matches[0].get("model_versions")
    v = _agent_data(run, "matching_agent", "model_versions")
    return v if isinstance(v, dict) else None
