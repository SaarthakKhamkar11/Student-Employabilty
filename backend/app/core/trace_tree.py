"""Rebuild one request's execution tree from what was persisted (spec §39, Phase 12; docs/adr/0020).

    get_agent_trace(request_id) -> TraceTree

    Supervisor
       +--> LLM routing call(s)            (agent_messages llm_call, if the LLM routed)
       +--> Profile Agent      (attempt 1)  (agent_runs row, parent_run_id = the request)
       |       +--> extract_text  #1        (agent_messages tool_call, one per attempt)
       |       ...
       +--> Matching Agent
       +--> Learning Agent     (skipped: reason)   (agent_messages skip)
       +--> Explainability Agent
               +--> generate_explanation #1
                       +--> LLM call         (nested by parent_span_id)

The tree is built only from the agent_runs and agent_messages rows the Supervisor wrote while the request ran:
every node is a real execution with its measured duration. A request that was never persisted has no trace
(None), never a reconstructed or example one. Phase 13's dashboard renders this.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.db.store import AgentRunStore, TraceRecords

NodeKind = Literal["supervisor", "agent", "tool", "llm", "skipped"]


class TraceNode(BaseModel):
    kind: NodeKind
    name: str
    id: str  # agent_runs.request_id for agents, the span id for the rest
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    attempt: int | None = None
    retry_count: int | None = None
    tool_call_count: int | None = None
    tool_latency_ms: float | None = None
    model_used: str | None = None
    model_version: str | None = None
    termination_reason: str | None = None
    error: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    children: list["TraceNode"] = Field(default_factory=list)


class TraceTree(BaseModel):
    request_id: str
    task_type: str
    status: str
    started_at: datetime
    finished_at: datetime
    duration_ms: float
    routing_source: str | None
    root: TraceNode
    counts: dict[str, int]  # agent executions, tool attempts, LLM calls, retries, skipped agents
    source: str = "agent_runs + agent_messages (persisted while the request ran)"


async def get_agent_trace(request_id: str, store: AgentRunStore | None = None) -> TraceTree | None:
    """The persisted trace of one request, or None if that request was never persisted."""
    if store is None:
        from app.core.config import get_settings
        from app.db.store import build_store

        store = build_store(get_settings())
        if store is None:
            raise RuntimeError("no database configured (DATABASE_URL): traces are read from agent_runs")
    records = await store.get_trace(request_id)
    return build_trace_tree(records) if records is not None else None


def build_trace_tree(records: TraceRecords) -> TraceTree:
    runs = {r["request_id"]: r for r in records.runs}
    root_row = records.runs[0]
    nodes = {rid: _run_node(r) for rid, r in runs.items()}
    root = nodes[root_row["request_id"]]
    root.kind = "supervisor"
    if root_row.get("routing_source") != "llm":  # "deterministic_router" names the fallback, not an LLM
        root.model_used = None
    for rid, r in runs.items():
        parent = r.get("parent_run_id")
        if parent in nodes and rid != root_row["request_id"]:
            nodes[parent].children.append(nodes[rid])
    # messages: nest under their parent span (an LLM call inside a tool attempt), else under their agent run
    spans = {}
    for m in sorted(records.messages, key=lambda m: (m["run_id"], m["sequence"])):
        spans[m["span_id"]] = (m, _message_node(m))
    for m, node in spans.values():
        parent = m.get("parent_span_id")
        if parent in spans:
            spans[parent][1].children.append(node)
        elif parent in nodes:
            nodes[parent].children.append(node)
        else:
            nodes[m["run_id"]].children.append(node)
    for node in list(nodes.values()) + [n for _, n in spans.values()]:
        node.children.sort(key=lambda n: (n.started_at, n.kind == "skipped"))
    kinds = [m["message_type"] for m in records.messages]
    counts = {"agent_executions": len(runs) - 1, "tool_attempts": kinds.count("tool_call"),
              "llm_calls": kinds.count("llm_call"), "skipped_agents": kinds.count("skip"),
              "agent_retries": sum(1 for r in records.runs[1:] if (r.get("attempt") or 1) > 1),
              "tool_retries": sum(1 for m in records.messages
                                  if m["message_type"] == "tool_call" and (m["content"].get("attempt") or 1) > 1)}
    return TraceTree(request_id=root_row["request_id"], task_type=root_row["task_type"], status=root_row["status"],
                     started_at=root_row["started_at"], finished_at=root_row["finished_at"],
                     duration_ms=root_row["duration_ms"], routing_source=root_row.get("routing_source"),
                     root=nodes[root_row["request_id"]], counts=counts)


def _run_node(r: dict[str, Any]) -> TraceNode:
    return TraceNode(kind="agent", name=r["agent_name"], id=r["request_id"], status=r["status"],
                     started_at=r["started_at"], finished_at=r["finished_at"], duration_ms=r["duration_ms"],
                     attempt=r.get("attempt"), retry_count=r.get("retry_count"),
                     tool_call_count=r.get("tool_call_count"), tool_latency_ms=r.get("tool_latency_ms"),
                     model_used=r.get("model_used"), model_version=r.get("model_version"),
                     termination_reason=r.get("termination_reason"), error=r.get("error_code"),
                     details={"model_versions": r.get("model_versions") or {}, "used_fallback": r.get("used_fallback")})


def _message_node(m: dict[str, Any]) -> TraceNode:
    kind = {"tool_call": "tool", "llm_call": "llm", "skip": "skipped"}.get(m["message_type"], "tool")
    content = m.get("content") or {}
    return TraceNode(kind=kind, name=m["recipient"], id=m["span_id"], status=m["status"],
                     started_at=m["started_at"], finished_at=m["finished_at"], duration_ms=m["duration_ms"],
                     attempt=content.get("attempt"), model_used=m["recipient"] if kind == "llm" else None,
                     error=" ".join(str(v) for v in (content.get("error_type"), content.get("http_status")) if v)
                     or None,
                     details={k: v for k, v in content.items() if k not in ("name", "attempt", "error_type")
                              and v is not None})


def render_trace(tree: TraceTree) -> str:
    """The tree as text, in the shape of spec §39/§40 (for the CLI, the demo and the report)."""
    lines = [f"request {tree.request_id}  task={tree.task_type}  status={tree.status}  "
             f"{tree.duration_ms:.0f} ms  routing={tree.routing_source}"]

    def label(n: TraceNode) -> str:
        if n.kind == "skipped":
            return f"{n.name}  [skipped: {n.details.get('reason')}]"
        bits = [n.name]
        if n.kind in ("agent", "supervisor"):
            bits.append(f"attempt {n.attempt}" if n.attempt and n.attempt > 1 else "")
            bits.append(f"{n.status}")
            bits.append(f"{n.duration_ms:.0f} ms")
            if n.tool_call_count:
                bits.append(f"{n.tool_call_count} tools / {n.tool_latency_ms:.0f} ms")
            if n.retry_count:
                what = "agent" if n.kind == "supervisor" else "tool"
                bits.append(f"{n.retry_count} {what} retr{'y' if n.retry_count == 1 else 'ies'}")
            if n.model_version:
                bits.append(f"model {n.model_version}")
            if n.model_used:
                bits.append(f"llm {n.model_used}")
            if n.error:
                bits.append(f"error {n.error}")
        elif n.kind == "tool":
            bits = [f"{n.name}" + (f" (attempt {n.attempt})" if n.attempt and n.attempt > 1 else ""),
                    "ok" if n.status == "success" else f"FAILED {n.error or ''}".strip(), f"{n.duration_ms:.1f} ms"]
        else:  # llm
            bits = [f"LLM {n.name} ({n.details.get('purpose')})",
                    "ok" if n.status == "success" else f"FAILED {n.error or ''}".strip(), f"{n.duration_ms:.0f} ms"]
        return "  ".join(b for b in bits if b)

    def walk(n: TraceNode, prefix: str) -> None:
        for i, child in enumerate(n.children):
            last = i == len(n.children) - 1
            lines.append(f"{prefix}+--> {label(child)}")
            walk(child, prefix + ("     " if last else "|    "))

    lines.append(label(tree.root))
    walk(tree.root, "")
    c = tree.counts
    lines.append(f"({c['agent_executions']} agent executions, {c['tool_attempts']} tool attempts, "
                 f"{c['llm_calls']} LLM calls, {c['skipped_agents']} skipped, {c['agent_retries']} agent retries, "
                 f"{c['tool_retries']} tool retries)")
    return "\n".join(lines)
