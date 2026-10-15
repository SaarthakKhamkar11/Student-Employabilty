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
      └─ persist the final AgentState to agent_runs (on failure persisted=false; never a crash), with the
         request's trace (Phase 12): one agent_runs row per agent execution, one agent_messages row per tool
         attempt, LLM call and skipped agent (app/core/tracing.py; rebuilt by app/core/trace_tree.py)

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
from app.core.logging import log_event, redact_text, register_sensitive_values, sensitive_values_var
from app.core.tracing import Span, TraceRecorder, current_span_id, current_trace, trace_var
from app.db.store import AgentMessageRecord, AgentRunRecord, AgentRunStore
from app.profile import pii as pii_detector
from app.llm import LLMClient
from app.schemas.agent_io import (
    AgentError, AgentOutcome, AgentOutput, CallTool, Delegate, ExplainabilityAgentInput, Finish,
    JobIntelligenceInput, LearningAgentInput, MatchingAgentInput, MatchResult, ProfileAgentInput,
    RoutingDecision, SkillGapReport, SupervisorInput, SupervisorOutput, SupervisorResponse, TaskType,
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

    def trace_model_versions(self, data: dict) -> dict[str, str]:
        routing = data.get("routing")
        return {"routing": routing["model_used"]} if isinstance(routing, dict) else {}

    def timeout_seconds(self) -> float:
        return self.settings.supervisor_timeout_seconds

    # -- public entrypoint -------------------------------------------------------
    async def handle(self, request: SupervisorInput | dict[str, Any], user_id: str | None = None) -> SupervisorResponse:
        started, t0 = _utcnow(), time.perf_counter()
        request = SupervisorInput.model_validate(request if isinstance(request, dict) else request.model_dump())
        state = AgentState(request_id=str(uuid.uuid4()), user_id=user_id, task_type="unclassified",
                           student_id=request.student_id, job_id=request.job_id, status="running")
        # Phase 12: this request's trace, and its PII registry. The resume text is scanned with the Phase 4
        # detector before anything runs, so even the first log line can't carry a detected value (a file upload
        # is scanned by the Profile Agent's detect_pii tool once its text is extracted).
        trace = TraceRecorder(state.request_id)
        tokens = trace_var.set(trace), sensitive_values_var.set(set())
        try:
            if request.resume_text:
                for f in pii_detector.detect(request.resume_text):
                    register_sensitive_values([request.resume_text[f.start:f.end]], words=f.type == "name")
            return await self._handle(request, state, trace, started, t0)
        finally:
            trace_var.reset(tokens[0])
            sensitive_values_var.reset(tokens[1])

    async def _handle(self, request: SupervisorInput, state: AgentState, trace: TraceRecorder, started: datetime,
                      t0: float) -> SupervisorResponse:
        output = await self.run(state, request)
        result = SupervisorOutput.model_validate(output.data) if output.status == "success" else None
        state.status = result.request_status if result else "failed"
        persisted = await self._persist(state, result, started, t0, trace)

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
            elif routing.fallback_reason != "llm_not_configured" and not (routing.fallback_reason or "").startswith(
                    "llm_not_needed"):
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
                if (trace := current_trace()) is not None:  # a decision, not an execution: zero duration
                    trace.event("skip", name, current_span_id(), "skipped", reason=skip_reason, wave=action.reason)
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
                if not (req.student_id or req.resume_provided):
                    return None, "no student_id or resume provided"
                return ProfileAgentInput(student_id=req.student_id, resume_text=req.resume_text,
                                         resume_file_b64=req.resume_file_b64,
                                         resume_filename=req.resume_filename), None
            if name == "job_intelligence_agent":
                if not (req.job_id or req.jd_text):
                    return None, "no job_id or job description provided"
                return JobIntelligenceInput(job_id=req.job_id, jd_text=req.jd_text), None
            if name == "matching_agent":
                student_id = req.student_id or _data_field(outcomes, "profile_agent", "profile", "student_id")
                job_id = req.job_id or _data_field(outcomes, "job_intelligence_agent", "posting", "job_id")
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
                report = top.skill_gap if isinstance(top.skill_gap, SkillGapReport) else None
                gaps = [g.skill for g in report.missing_required + report.missing_preferred] if report else []
                if not gaps:
                    return None, NOT_NEEDED + "no skill gaps were identified"
                return LearningAgentInput(student_id=top.student_id, job_id=top.job_id, skill_gaps=gaps,
                                          skill_gap_report=report, job_title=top.summary.get("job_title")), None
            if name == "explainability_agent":
                return ExplainabilityAgentInput(match=top), None
        except ValidationError as exc:
            return None, f"could not build a valid input ({exc.error_count()} error(s))"
        return None, "no input builder for this agent"

    # -- persistence -------------------------------------------------------------
    async def _persist(self, state: AgentState, result: SupervisorOutput | None, started: datetime, t0: float,
                       trace: TraceRecorder) -> bool:
        if self.store is None:
            log_event(self.logger, "persistence_skipped", logging.WARNING, reason="no database configured")
            return False
        root = trace.get(state.request_id)
        attrs = root.attrs if root is not None else {}
        record = AgentRunRecord(
            request_id=state.request_id, user_id=state.user_id, task_type=state.task_type,
            student_id=state.student_id, job_id=state.job_id, status=state.status,
            routing_source=result.routing.source if result else None,
            model_used=result.routing.model_used if result else None,
            agents_called=state.agents_called, agent_results=state.agent_results,
            model_versions=state.model_versions, errors=state.errors,
            started_at=started, finished_at=_utcnow(), duration_ms=_ms(t0),
            agent_name=self.name, termination_reason=attrs.get("termination_reason"),
            # the Supervisor's own retries are its re-runs of agents (each a second execution row)
            retry_count=sum(1 for s in trace.spans if s.kind == "agent" and s.attrs.get("attempt", 1) > 1),
            used_fallback=bool(result and result.routing.source != "llm"),
            model_version=attrs.get("model_version"),
            error_code=attrs.get("error"),
        )
        executions, messages = _trace_rows(state, trace)
        try:
            await asyncio.wait_for(self.store.save(record, executions, messages), timeout=10)
            return True
        except Exception as exc:  # a database failure must not break the request (spec §21)
            state.errors.append(_error("persistence_failed", type(exc).__name__))
            log_event(self.logger, "persistence_failed", logging.ERROR, error=type(exc).__name__)
            return False


def _trace_rows(state: AgentState, trace: TraceRecorder) -> tuple[list[AgentRunRecord], list[AgentMessageRecord]]:
    """The recorded spans as rows: every agent execution below the Supervisor becomes an agent_runs row, and every
    other span (tool attempt, LLM call, skip) an agent_messages row of the agent execution it happened in."""
    executions = []
    for span in trace.spans:
        if span.kind != "agent" or span.span_id == state.request_id or span.finished_at is None:
            continue
        a = span.attrs
        owner = trace.owning_agent(span)
        executions.append(AgentRunRecord(
            request_id=span.span_id, parent_run_id=owner.span_id if owner else state.request_id,
            user_id=state.user_id, task_type=a.get("task") or state.task_type, student_id=state.student_id,
            job_id=state.job_id, status=span.status, routing_source=None, model_used=a.get("model_used"),
            agents_called=[span.name], agent_results={}, model_versions=a.get("model_versions") or {},
            errors=a.get("errors") or [], started_at=span.started_at, finished_at=span.finished_at,
            duration_ms=span.duration_ms, agent_name=span.name, attempt=a.get("attempt", 1),
            termination_reason=a.get("termination_reason"), retry_count=a.get("retry_count", 0),
            used_fallback=bool(a.get("fallback_used")), tool_call_count=len(a.get("tool_calls") or []),
            tool_latency_ms=a.get("tool_latency_ms", 0.0), model_version=a.get("model_version"),
            error_code=a.get("error")))
    messages, sequence = [], {}
    for span in trace.spans:
        if span.kind == "agent" or span.finished_at is None:
            continue
        owner = trace.owning_agent(span)
        run_id = owner.span_id if owner else state.request_id
        n = sequence[run_id] = sequence.get(run_id, -1) + 1
        messages.append(AgentMessageRecord(
            run_id=run_id, sequence=n, sender=owner.name if owner else "supervisor", recipient=span.name,
            message_type={"tool": "tool_call", "llm": "llm_call", "skip": "skip"}[span.kind], span_id=span.span_id,
            parent_span_id=span.parent_id, status=span.status, started_at=span.started_at,
            finished_at=span.finished_at, duration_ms=span.duration_ms, content=_message_content(span)))
    return executions, messages


def _message_content(span: Span) -> dict[str, Any]:
    keep = {"tool": ("attempt", "error_type"), "llm": ("purpose", "error_type", "http_status", "agent_name"),
            "skip": ("reason", "wave")}[span.kind]
    return {"name": span.name, **{k: span.attrs.get(k) for k in keep}}


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
