"""Per-request execution trace (spec §39, Phase 12; docs/adr/0020).

The Supervisor opens one TraceRecorder per request. The runtime then records a span for each thing that actually
runs, at the moment it runs:

    agent  BaseAgent.run(): one span per execution. A Supervisor retry is a second span (attempt 2)
    tool   BaseAgent._call_tool_with_retry(): one span per attempt, so retries are visible
    llm    TracedLLMClient (app/llm/client.py): one span per provider call (latency, model, HTTP status)
    skip   the Supervisor decided not to run a planned agent (zero duration; the reason is recorded)

Parents come from a context variable, so asyncio.gather() (parallel agent waves) and asyncio.to_thread()
(synchronous tools) nest each span under whatever was running when it started. Nothing here is synthetic:
a span exists only because the runtime started it, and its duration is measured, never estimated.

At the end of the request the Supervisor persists the spans: agent spans as agent_runs rows (parent_run_id), the
others as agent_messages rows (app/db/store.py). app/core/trace_tree.py rebuilds the tree from those rows.
"""

import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Literal

SpanKind = Literal["agent", "tool", "llm", "skip"]


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    kind: SpanKind
    name: str
    started_at: datetime
    sequence: int  # start order within the request
    finished_at: datetime | None = None
    duration_ms: float | None = None
    status: str = "running"
    attrs: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default=0.0, repr=False)


class TraceRecorder:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.spans: list[Span] = []
        self._lock = threading.Lock()  # tools run in worker threads

    def start(self, kind: SpanKind, name: str, parent_id: str | None, span_id: str | None = None,
              **attrs: Any) -> Span:
        with self._lock:
            span = Span(span_id=span_id or str(uuid.uuid4()), parent_id=parent_id, kind=kind, name=name,
                        started_at=_utcnow(), sequence=len(self.spans), attrs=dict(attrs), _t0=time.perf_counter())
            self.spans.append(span)
        return span

    def finish(self, span: Span, status: str, **attrs: Any) -> None:
        span.finished_at = _utcnow()
        span.duration_ms = round((time.perf_counter() - span._t0) * 1000, 3)
        span.status = status
        span.attrs.update(attrs)

    def event(self, kind: SpanKind, name: str, parent_id: str | None, status: str, **attrs: Any) -> Span:
        """Something decided rather than run (an agent the Supervisor skipped): zero duration."""
        span = self.start(kind, name, parent_id, **attrs)
        span._t0 = time.perf_counter()
        self.finish(span, status)
        span.duration_ms = 0.0
        return span

    # -- queries ---------------------------------------------------------------
    def get(self, span_id: str) -> Span | None:
        return next((s for s in self.spans if s.span_id == span_id), None)

    def children(self, span_id: str) -> list[Span]:
        return [s for s in self.spans if s.parent_id == span_id]

    def descendants(self, span_id: str, kind: SpanKind | None = None) -> list[Span]:
        out, todo = [], [span_id]
        while todo:
            for child in self.children(todo.pop()):
                if kind is None or child.kind == kind:
                    out.append(child)
                if child.kind != "agent":  # an agent's own subtree is its own
                    todo.append(child.span_id)
        return out

    def owning_agent(self, span: Span) -> Span | None:
        """The nearest agent span above span (its agent_runs row)."""
        parent = self.get(span.parent_id) if span.parent_id else None
        while parent is not None and parent.kind != "agent":
            parent = self.get(parent.parent_id) if parent.parent_id else None
        return parent

    def attempt_of(self, name: str, parent_id: str | None) -> int:
        return sum(1 for s in self.spans if s.kind == "agent" and s.name == name and s.parent_id == parent_id)


trace_var: ContextVar[TraceRecorder | None] = ContextVar("trace", default=None)
span_var: ContextVar[str | None] = ContextVar("span", default=None)


def current_trace() -> TraceRecorder | None:
    return trace_var.get()


def current_span_id() -> str | None:
    return span_var.get()


@contextmanager
def open_span(kind: SpanKind, name: str, span_id: str | None = None, **attrs: Any) -> Iterator[Span | None]:
    """Record a span around a block, as the child of the current span. A no-op (yields None) outside a request.

    The block may finish the span itself (to set its status and attributes); otherwise it is finished here,
    "failed" if the block raised.
    """
    trace = trace_var.get()
    if trace is None:
        yield None
        return
    span = trace.start(kind, name, span_var.get(), span_id=span_id, **attrs)
    token = span_var.set(span.span_id)
    try:
        yield span
    except BaseException as exc:
        if span.finished_at is None:
            trace.finish(span, "failed", error_type=type(exc).__name__)
        raise
    finally:
        span_var.reset(token)
    if span.finished_at is None:
        trace.finish(span, "success")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
