"""Supervisor Agent (spec §5.1), implemented in Phase 3.

One request, end to end:

    handle(request)
      ├─ run()  (the BaseAgent loop: validated decisions, timeout, tracing)
      │    ├─ decision 1: route. The LLM picks the agents (validated), or the
      │    │   deterministic fallback router does (see routing.py).
      │    ├─ Delegate wave 1: profile_agent ‖ job_intelligence_agent   (only those planned)
      │    ├─ Delegate wave 2: matching_agent
      │    ├─ Delegate wave 3: learning_agent ‖ explainability_agent
      │    └─ Finish → finalize(): aggregate the outcomes; nothing is invented
      └─ persist the final AgentState to agent_runs (on failure persisted=false; never a crash)

The LLM only routes. It never sees resume or JD bodies, and never computes anything.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal, get_args

from pydantic import BaseModel, ValidationError

from app.agents.base import GUARDRAILS, AgentConfigurationError, AgentContext, BaseAgent
from app.agents.routing import EXECUTION_WAVES, SPECIALISTS, UPSTREAM, Router
from app.core.config import Settings
from app.core.logging import log_event, redact_text
from app.db.store import AgentRunRecord, AgentRunStore
from app.llm import LLMClient
from app.schemas.agent_io import (
    AgentError, AgentOutcome, AgentOutput, CallTool, Delegate, ExplainabilityAgentInput, Finish,
    JobIntelligenceInput, LearningAgentInput, MatchingAgentInput, MatchResult, ProfileAgentInput,
    RoutingDecision, SkillGap, SupervisorInput, SupervisorOutput, SupervisorResponse, TaskType,
)
from app.schemas.agent_state import INSUFFICIENT_EVIDENCE, AgentState
from app.tools.registry import ToolRegistry

TRANSIENT_FAILURES = {"timeout", "tool_failure", "internal_error"}
NOT_NEEDED = "not needed: "  # prefix for skips that are not failures


@dataclass(frozen=True)
class FailurePolicy:
    retry_once: bool
    on_failure: Literal["fail_request", "degrade"]


# What the Supervisor does when a specialist fails. Only transient failures
# (timeout, tool_failure, internal_error) are retried, and only once: a
# deterministic failure such as not_implemented or invalid_input would fail
# the same way again. A failed or skipped agent's result is always reported
# as a sentinel, never papered over.
FAILURE_POLICY: dict[str, FailurePolicy] = {
    # Without a structured student profile there is nothing to match: fail the request.
    "profile_agent": FailurePolicy(retry_once=True, on_failure="fail_request"),
    # Without a structured job, a company request has nothing to rank against: fail the request.
    "job_intelligence_agent": FailurePolicy(retry_once=True, on_failure="fail_request"),
    # Matching is the core result, and learning and explanations depend on it: fail the request.
    "matching_agent": FailurePolicy(retry_once=True, on_failure="fail_request"),
    # Enrichments: still return the match results without them, with request status "partial".
    "learning_agent": FailurePolicy(retry_once=True, on_failure="degrade"),
    "explainability_agent": FailurePolicy(retry_once=True, on_failure="degrade"),
}


class SupervisorAgent(BaseAgent):
    name = "supervisor"
    role_description = (
        "You are the Supervisor Agent of a student employability and job-matching system. Identify "
        "the task type of each request, decide which specialist agents it actually needs (never all "
        "of them by default), create an execution plan, run it, validate every agent result, retry or "
        "fall back on failure, and aggregate the results into one structured response. Maintain the "
        "workflow state; never invent a missing result. "
        + GUARDRAILS
    )
    allowed_tools: ClassVar[list[str]] = []  # the Supervisor delegates to agents instead of calling tools
    allowed_delegates: ClassVar[list[str]] = list(SPECIALISTS)
    # Spec §20 Supervisor tools map to: route_agent = routing step, validate_result = _validated(),
    # retry_agent = FAILURE_POLICY, aggregate_results = finalize().
    planned_tools = ("route_agent", "validate_result", "retry_agent", "aggregate_results")
    input_schema = SupervisorInput
    output_schema = SupervisorOutput
    implemented = True
    implemented_in_phase = 3

    def __init__(self, agents: dict[str, BaseAgent], llm: LLMClient | None = None,
                 store: AgentRunStore | None = None, registry: ToolRegistry | None = None,
                 settings: Settings | None = None):
        super().__init__(registry, settings)
        missing = [a for a in SPECIALISTS if a not in agents]
        if missing:
            raise AgentConfigurationError(f"supervisor: missing specialist agents {missing}")
        self.agents = agents
        self.router = Router(llm, self.settings)
        self.store = store

    def timeout_seconds(self) -> float:
        return self.settings.supervisor_timeout_seconds

    # -- public entrypoint -------------------------------------------------------
    async def handle(self, request: SupervisorInput | dict[str, Any], user_id: str | None = None) -> SupervisorResponse:
        started, t0 = _utcnow(), time.perf_counter()
        request = SupervisorInput.model_validate(request if isinstance(request, dict) else request.model_dump())
        state = AgentState(request_id=str(uuid.uuid4()), user_id=user_id, task_type="unclassified",
                           student_id=request.student_id, job_id=request.job_id, status="running")

        output = await self.run(state, request)
        result = SupervisorOutput.model_validate(output.data) if output.status == "success" else None
        state.status = result.request_status if result else "failed"
        persisted = await self._persist(state, result, started, t0)

        task_type = state.task_type if state.task_type in get_args(TaskType) else "unknown"
        summary = result.summary if result else (
            f"The Supervisor could not complete the request ({output.termination_reason}). "
            "No results were produced.")
        response = SupervisorResponse(
            request_id=state.request_id, status=state.status, task_type=task_type,
            routing=result.routing if result else INSUFFICIENT_EVIDENCE,
            results=result.outcomes if result else {}, summary=summary,
            model_versions=state.model_versions, errors=state.errors, persisted=persisted,
            started_at=started, finished_at=_utcnow(), duration_ms=_ms(t0),
        )
        log_event(self.logger, "request_complete", agent_name=self.name, task=task_type, status=response.status,
                  agents_called=state.agents_called, persisted=persisted, duration_ms=response.duration_ms)
        return response

    # -- decisions ---------------------------------------------------------------
    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Delegate | Finish:
        if "routing" not in ctx.extras:
            routing = await self.router.route(ctx.input)
            ctx.extras.update(routing=routing, outcomes={},
                              pending_waves=[w for w in ([a for a in wave if a in routing.agents]
                                                         for wave in EXECUTION_WAVES) if w])
            ctx.state.task_type = routing.task_type
            if routing.source == "llm":
                ctx.model_used = routing.model_used
            elif routing.fallback_reason != "llm_not_configured":
                ctx.state.errors.append(_error("llm_routing_fallback", routing.fallback_reason or ""))
        if ctx.extras["pending_waves"]:
            wave = ctx.extras["pending_waves"].pop(0)
            return Delegate(agents=wave, reason="execution wave: " + ", ".join(wave))
        return Finish(reason="all planned agents handled")

    async def delegate(self, ctx: AgentContext, action: Delegate) -> None:
        outcomes: dict[str, AgentOutcome] = ctx.extras["outcomes"]
        runnable = []
        for name in action.agents:
            payload, skip_reason = self._build_input(name, ctx)
            if skip_reason:
                outcomes[name] = AgentOutcome(agent_name=name, status="skipped", attempts=0,
                                              data=INSUFFICIENT_EVIDENCE, skip_reason=skip_reason)
                log_event(self.logger, "agent_skipped", agent_name=name, reason=skip_reason)
            else:
                runnable.append((name, payload))
        results = await asyncio.gather(*(self._run_with_policy(ctx.state, n, p) for n, p in runnable))
        outcomes.update({name: outcome for (name, _), outcome in zip(runnable, results)})

    async def finalize(self, ctx: AgentContext) -> SupervisorOutput:
        routing: RoutingDecision = ctx.extras["routing"]
        outcomes = {a: ctx.extras["outcomes"].get(a) or AgentOutcome(
            agent_name=a, status="skipped", attempts=0, data=INSUFFICIENT_EVIDENCE, skip_reason="not reached")
            for a in routing.agents}
        status = _request_status(routing, outcomes)
        return SupervisorOutput(task_type=routing.task_type, request_status=status, routing=routing,
                                outcomes=outcomes, summary=_summary(routing, outcomes, status))

    # -- running specialists -----------------------------------------------------
    async def _run_with_policy(self, state: AgentState, name: str, payload: BaseModel) -> AgentOutcome:
        agent, policy = self.agents[name], FAILURE_POLICY[name]
        attempts = 0
        while True:
            attempts += 1
            output = self._validated(agent, await agent.run(state, payload), state)
            if output.status == "success" or not (policy.retry_once and attempts == 1
                                                   and output.termination_reason in TRANSIENT_FAILURES):
                break
            log_event(self.logger, "agent_retry", logging.WARNING, agent_name=name, attempt=attempts,
                      termination_reason=output.termination_reason)
        return AgentOutcome(agent_name=name, status=output.status, attempts=attempts, data=output.data,
                            termination_reason=output.termination_reason,
                            error_codes=[e.code for e in output.errors])

    def _validated(self, agent: BaseAgent, output: Any, state: AgentState) -> AgentOutput:
        """Re-validate an agent's envelope and, on success, its data against its output schema."""
        try:
            if not isinstance(output, AgentOutput):
                raise TypeError(f"run() returned {type(output).__name__}")
            output = AgentOutput.model_validate(output.model_dump())
            if output.status == "success":
                agent.output_schema.model_validate(output.data)
            return output
        except (ValidationError, TypeError):
            error = AgentError(agent_name=self.name, code="invalid_agent_output",
                               message=f"{agent.name} returned output that failed validation", timestamp=_utcnow())
            state.errors.append(error.model_dump(mode="json"))
            now = _utcnow()
            return AgentOutput(agent_name=agent.name, status="failed", data=INSUFFICIENT_EVIDENCE, errors=[error],
                               termination_reason="invalid_output", started_at=now, finished_at=now, duration_ms=0.0)

    def _build_input(self, name: str, ctx: AgentContext) -> tuple[BaseModel | None, str | None]:
        """Build an agent's structured input from the request and upstream outcomes, or say why it can't run."""
        req: SupervisorInput = ctx.input
        routing: RoutingDecision = ctx.extras["routing"]
        outcomes: dict[str, AgentOutcome] = ctx.extras["outcomes"]
        for upstream in UPSTREAM.get(name, ()):
            done = outcomes.get(upstream)
            if upstream in routing.agents and (done is None or done.status != "success"):
                return None, f"upstream {upstream} {done.status if done else 'did not run'}"
        try:
            if name == "profile_agent":
                if not (req.student_id or req.resume_text):
                    return None, "no student_id or resume provided"
                return ProfileAgentInput(student_id=req.student_id, resume_text=req.resume_text), None
            if name == "job_intelligence_agent":
                if not (req.job_id or req.jd_text):
                    return None, "no job_id or job description provided"
                return JobIntelligenceInput(job_id=req.job_id, jd_text=req.jd_text), None
            if name == "matching_agent":
                student_id = req.student_id or _data_field(outcomes, "profile_agent", "student", "student_id")
                job_id = req.job_id or _data_field(outcomes, "job_intelligence_agent", "job", "job_id")
                if req.actor == "company" or routing.task_type in ("rank_candidates_for_job", "ingest_job"):
                    if not job_id:
                        return None, "no job_id available for candidate matching"
                    return MatchingAgentInput(mode="candidates_for_job", job_id=job_id, student_id=student_id), None
                if not student_id:
                    return None, "no student_id available for job matching"
                return MatchingAgentInput(mode="jobs_for_student", student_id=student_id, job_id=job_id), None
            top = _top_match(outcomes)
            if top is None:
                return None, "no match result available"
            if name == "learning_agent":
                gaps = top.skill_gap.missing_skills if isinstance(top.skill_gap, SkillGap) else []
                if not gaps:
                    return None, NOT_NEEDED + "no skill gaps were identified"
                return LearningAgentInput(student_id=top.student_id, job_id=top.job_id, skill_gaps=gaps), None
            if name == "explainability_agent":
                return ExplainabilityAgentInput(match=top), None
        except ValidationError as exc:
            return None, f"could not build a valid input ({exc.error_count()} error(s))"
        return None, "no input builder for this agent"

    # -- persistence -------------------------------------------------------------
    async def _persist(self, state: AgentState, result: SupervisorOutput | None, started: datetime, t0: float) -> bool:
        if self.store is None:
            log_event(self.logger, "persistence_skipped", logging.WARNING, reason="no database configured")
            return False
        record = AgentRunRecord(
            request_id=state.request_id, user_id=state.user_id, task_type=state.task_type,
            student_id=state.student_id, job_id=state.job_id, status=state.status,
            routing_source=result.routing.source if result else None,
            model_used=result.routing.model_used if result else None,
            agents_called=state.agents_called, agent_results=state.agent_results,
            model_versions=state.model_versions, errors=state.errors,
            started_at=started, finished_at=_utcnow(), duration_ms=_ms(t0),
        )
        try:
            await asyncio.wait_for(self.store.save(record), timeout=10)
            return True
        except Exception as exc:  # a database failure must not break the request (spec §21)
            state.errors.append(_error("persistence_failed", type(exc).__name__))
            log_event(self.logger, "persistence_failed", logging.ERROR, error=type(exc).__name__)
            return False


def _request_status(routing: RoutingDecision, outcomes: dict[str, AgentOutcome]) -> str:
    if not routing.agents:
        return "failed"

    def failed(o: AgentOutcome) -> bool:
        return o.status in ("failed", "partial") or (o.status == "skipped" and not (o.skip_reason or "").startswith(NOT_NEEDED))

    if any(failed(o) and FAILURE_POLICY[a].on_failure == "fail_request" for a, o in outcomes.items()):
        return "failed"
    return "partial" if any(failed(o) for o in outcomes.values()) else "success"


def _summary(routing: RoutingDecision, outcomes: dict[str, AgentOutcome], status: str) -> str:
    """Deterministic summary of what happened. Phase 11 adds evidence-based natural-language explanations."""
    if not routing.agents:
        return ("The request could not be mapped to a supported task, so no agent was run. "
                "Insufficient evidence to answer it.")
    router = (f"the LLM ({routing.model_used})" if routing.source == "llm"
              else f"the deterministic fallback router ({routing.fallback_reason})")
    lines = [f"Task '{routing.task_type}' was routed by {router}."]
    for name, o in outcomes.items():
        detail = f" ({o.termination_reason})" if o.status in ("failed", "partial") else ""
        detail += f": {o.skip_reason}" if o.status == "skipped" else ""
        lines.append(f"{name}: {o.status}{detail}.")
    lines.append("Results that were not produced are reported as 'Not yet evaluated' or 'Insufficient evidence', "
                 f"never estimated. Request status: {status}.")
    return " ".join(lines)


def _data_field(outcomes: dict[str, AgentOutcome], agent: str, obj: str, field: str) -> str | None:
    done = outcomes.get(agent)
    if done is None or done.status != "success" or not isinstance(done.data, dict):
        return None
    value = done.data.get(obj)
    return value.get(field) if isinstance(value, dict) else None


def _top_match(outcomes: dict[str, AgentOutcome]) -> MatchResult | None:
    done = outcomes.get("matching_agent")
    if done is None or done.status != "success" or not isinstance(done.data, dict):
        return None
    matches = done.data.get("matches")
    if not isinstance(matches, list) or not matches:
        return None
    try:
        return MatchResult.model_validate(min(matches, key=lambda m: m.get("rank", 10**9)))
    except (ValidationError, AttributeError):
        return None


def _error(code: str, message: str) -> dict[str, Any]:
    return AgentError(agent_name="supervisor", code=code, message=redact_text(message),
                      timestamp=_utcnow()).model_dump(mode="json")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)
