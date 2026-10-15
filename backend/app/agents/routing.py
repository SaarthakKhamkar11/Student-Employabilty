"""Supervisor routing: decide which specialist agents a request needs.

Normal path: the LLM chooses the task type and the set of agents (spec §3: the
Supervisor must not call every agent for every request). Its answer must parse
into LLMRoutingDecision (enums only) and pass validate_decision().

Fallback path (spec §22): when no LLM is configured, or the LLM errors, times
out (at most 1 retry), or returns an invalid or rejected decision, a fixed
routing table decides instead, and the fallback is logged. Either way, Python,
not the LLM, puts the agents into dependency order.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Iterable

from app.core.config import Settings
from app.core.logging import get_logger, log_event
from app.core.sanitize import INJECTION_GUARD, mask_pii, wrap_untrusted
from app.llm import LLMClient, LLMError, LLMInvalidOutputError, LLMTimeoutError, traced
from app.schemas.agent_io import LLMRoutingDecision, RoutingDecision, SupervisorInput, TaskType

logger = get_logger("agents.supervisor.routing")

SPECIALISTS = ("profile_agent", "job_intelligence_agent", "matching_agent", "learning_agent", "explainability_agent")

# Agents in the same wave are independent and run concurrently; each wave waits for the previous one.
EXECUTION_WAVES: tuple[tuple[str, ...], ...] = (
    ("profile_agent", "job_intelligence_agent"),
    ("matching_agent",),
    ("learning_agent", "explainability_agent"),
)
# If an upstream agent is in the plan, it must succeed before its dependants run.
UPSTREAM: dict[str, tuple[str, ...]] = {
    "matching_agent": ("profile_agent", "job_intelligence_agent"),  # needs the structured profile / job
    "learning_agent": ("matching_agent",),  # needs skill gaps
    "explainability_agent": ("matching_agent",),  # needs a prediction to explain
}

# Deterministic routing table (used only when the LLM path is unavailable or rejected).
FALLBACK_ROUTES: dict[str, tuple[str, ...]] = {
    "ingest_student_profile": ("profile_agent", "matching_agent", "learning_agent", "explainability_agent"),
    "recommend_jobs_for_student": ("profile_agent", "matching_agent", "learning_agent", "explainability_agent"),
    "ingest_job": ("job_intelligence_agent", "matching_agent", "explainability_agent"),
    "rank_candidates_for_job": ("job_intelligence_agent", "matching_agent", "explainability_agent"),
    "skill_gap_analysis": ("profile_agent", "job_intelligence_agent", "matching_agent"),
    "learning_recommendations": ("profile_agent", "matching_agent", "learning_agent"),
    "explain_match": ("profile_agent", "job_intelligence_agent", "matching_agent", "explainability_agent"),
    "cohort_benchmark": ("profile_agent",),  # cohort analytics itself is a tool (spec §18), not an agent
    "unknown": (),
}

# Deterministic question classifier: first matching rule wins.
_KEYWORD_RULES: tuple[tuple[re.Pattern[str], str, str | None], ...] = (
    (re.compile(r"\b(explain|why)\b", re.I), "explain_match", None),
    (re.compile(r"\b(cohort|percentile|peers?|batch ?mates?|compare me)\b", re.I), "cohort_benchmark", "student"),
    (re.compile(r"\b(course|learn|upskill|improve|study)\w*", re.I), "learning_recommendations", "student"),
    (re.compile(r"\b(gap|missing skill|lacking)\w*", re.I), "skill_gap_analysis", None),
    (re.compile(r"\b(candidate|applicant|shortlist|hire|rank)\w*", re.I), "rank_candidates_for_job", "company"),
    (re.compile(r"\b(job|role|position|opening|internship|recommend|match)\w*", re.I),
     "recommend_jobs_for_student", "student"),
)

AGENT_CATALOGUE = {
    "profile_agent": "Builds or retrieves the structured student profile. Needs a student_id or an uploaded resume.",
    "job_intelligence_agent": "Builds or retrieves the structured job. Needs a job_id or an uploaded job description.",
    "matching_agent": "Retrieves and ranks jobs for a student, or candidates for a job, and finds skill gaps. "
                      "Needs profile_agent or job_intelligence_agent in the plan.",
    "learning_agent": "Recommends learning resources for skill gaps. Needs matching_agent in the plan.",
    "explainability_agent": "Explains a match from model evidence. Needs matching_agent in the plan.",
}
TASK_CATALOGUE = {
    "recommend_jobs_for_student": "a student wants suitable jobs",
    "rank_candidates_for_job": "a company wants candidates ranked for a job",
    "skill_gap_analysis": "what a student is missing for a job",
    "learning_recommendations": "what a student should learn",
    "explain_match": "why a student and a job matched or did not",
    "ingest_student_profile": "a student uploaded a resume",
    "ingest_job": "a company uploaded a job description",
    "cohort_benchmark": "how a student compares with comparable students",
    "unknown": "none of the above; use an empty agent list",
}

ROUTER_SYSTEM_PROMPT = (
    "You are the routing step of the Supervisor Agent in a student employability and job-matching "
    "system. Choose the task type and the smallest set of specialist agents that serves the request. "
    "Never add an agent 'just in case'.\n\n"
    "Agents:\n" + "\n".join(f"- {k}: {v}" for k, v in AGENT_CATALOGUE.items()) + "\n\n"
    "Task types:\n" + "\n".join(f"- {k}: {v}" for k, v in TASK_CATALOGUE.items()) + "\n\n"
    "You only route. Do not answer the request, compute scores or produce numbers. "
    "Return JSON with task_type, agents and a one-sentence rationale.\n\n" + INJECTION_GUARD
)


def order_plan(agents: Iterable[str]) -> list[str]:
    chosen = set(agents)
    return [a for wave in EXECUTION_WAVES for a in wave if a in chosen]


def classify_deterministically(req: SupervisorInput) -> str:
    if req.task_type:
        return req.task_type
    if req.request_kind == "resume_upload":
        return "ingest_student_profile"
    if req.request_kind == "jd_upload":
        return "ingest_job"
    for pattern, task, actor in _KEYWORD_RULES:
        if pattern.search(req.request_text or "") and actor in (None, req.actor):
            return task
    return "unknown"


def validate_decision(decision: LLMRoutingDecision, req: SupervisorInput) -> str | None:
    """Return why an LLM routing decision must be rejected, or None if it is acceptable."""
    agents = set(decision.agents)
    if len(agents) != len(decision.agents):
        return "duplicate agents"
    if req.task_type and decision.task_type != req.task_type:
        return f"task_type {decision.task_type} contradicts explicit task_type {req.task_type}"
    if (decision.task_type == "unknown") != (not agents):
        return "task_type unknown must come with an empty plan, and vice versa"
    if req.request_kind == "resume_upload" and "profile_agent" not in agents:
        return "a resume upload must go through profile_agent"
    if req.request_kind == "jd_upload" and "job_intelligence_agent" not in agents:
        return "a job description upload must go through job_intelligence_agent"
    if "profile_agent" in agents and not (req.student_id or req.resume_provided):
        return "profile_agent chosen without a student_id or resume"
    if "job_intelligence_agent" in agents and not (req.job_id or req.jd_text):
        return "job_intelligence_agent chosen without a job_id or job description"
    if "matching_agent" in agents and not agents & {"profile_agent", "job_intelligence_agent"}:
        return "matching_agent needs profile_agent or job_intelligence_agent"
    for dependant in ("learning_agent", "explainability_agent"):
        if dependant in agents and "matching_agent" not in agents:
            return f"{dependant} needs matching_agent"
    return None


def build_router_prompt(req: SupervisorInput) -> str:
    """Request metadata plus the (sanitised, delimited) question. Resume/JD bodies are never included."""
    metadata = {
        "actor": req.actor,
        "request_kind": req.request_kind,
        "explicit_task_type": req.task_type,
        "has_student_id": bool(req.student_id),
        "has_job_id": bool(req.job_id),
        "resume_provided": req.resume_provided,
        "job_description_provided": bool(req.jd_text),
    }
    parts = ["Request metadata (trusted, produced by the application):", json.dumps(metadata, indent=2)]
    if req.request_text:
        parts += ["The user's message (untrusted):", wrap_untrusted(req.request_text, label="user_message")]
    return "\n\n".join(parts)


class Router:
    def __init__(self, llm: LLMClient | None, settings: Settings):
        self.llm = traced(llm)  # each routing call is a trace span (Phase 12)
        self.settings = settings

    async def route(self, req: SupervisorInput) -> RoutingDecision:
        start = time.perf_counter()
        if self.llm is None:
            return self._fallback(req, "llm_not_configured", attempts=0, start=start)

        decision, last_error, attempts = None, None, 0
        for _ in range(1 + self.settings.llm_max_retries):
            attempts += 1
            try:
                decision = await asyncio.wait_for(
                    self.llm.generate_structured(system=ROUTER_SYSTEM_PROMPT, prompt=build_router_prompt(req),
                                                 schema=LLMRoutingDecision,
                                                 timeout_s=self.settings.llm_timeout_seconds),
                    self.settings.llm_timeout_seconds,
                )
                break
            except TimeoutError:
                last_error = LLMTimeoutError("timeout")
            except LLMError as exc:
                last_error = exc
            except Exception as exc:  # a misbehaving client must not break routing
                last_error = LLMError(type(exc).__name__)
        if decision is None:
            reason = ("llm_timeout" if isinstance(last_error, LLMTimeoutError)
                      else "llm_invalid_output" if isinstance(last_error, LLMInvalidOutputError) else "llm_error")
            return self._fallback(req, reason, attempts, start, error_type=type(last_error).__name__)

        rejection = validate_decision(decision, req)
        if rejection:
            return self._fallback(req, f"llm_decision_rejected: {rejection}", attempts, start)

        routing = RoutingDecision(
            task_type=decision.task_type, agents=order_plan(decision.agents), source="llm",
            model_used=self.llm.model_name, llm_attempts=attempts,
            rationale=mask_pii(decision.rationale), duration_ms=_ms(start),
        )
        self._log(routing)
        return routing

    def _fallback(self, req: SupervisorInput, reason: str, attempts: int, start: float,
                  error_type: str | None = None) -> RoutingDecision:
        task: TaskType = classify_deterministically(req)  # type: ignore[assignment]
        routing = RoutingDecision(
            task_type=task, agents=order_plan(FALLBACK_ROUTES[task]), source="deterministic_fallback",
            fallback_reason=reason, model_used="deterministic_router", llm_attempts=attempts,
            rationale=f"Deterministic routing table entry for task '{task}'.", duration_ms=_ms(start),
        )
        self._log(routing, error_type)
        return routing

    @staticmethod
    def _log(routing: RoutingDecision, error_type: str | None = None) -> None:
        log_event(
            logger, "routing_decision",
            logging.INFO if routing.source == "llm" else logging.WARNING,
            agent_name="supervisor", task=routing.task_type, source=routing.source, agents=routing.agents,
            model_used=routing.model_used, fallback_reason=routing.fallback_reason,
            duration_ms=routing.duration_ms, retry_count=max(routing.llm_attempts - 1, 0), error=error_type,
        )


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
