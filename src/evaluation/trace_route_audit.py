"""Phase 14, RQ9 evidence: did the Supervisor run only the agents each request type needs? (docs/adr/0021)

    python -m src.evaluation.trace_route_audit        (DATABASE_URL: read-only queries on agent_runs / agent_messages)

For every persisted request that has execution children (Phase 12 traces), compare
  called    agents_called on the request row (the agents the Supervisor called; skipped ones are not listed)
  expected  FALLBACK_ROUTES[task_type] (app/agents/routing.py: the agents a task needs, fixed in Phase 3)
  executed  child agent_runs rows (agents that actually ran)
  skipped   agent_messages of type 'skip' (agents the Supervisor decided not to run, with the reason)
and summarise per task type: agents never run for that task, skips and their reasons, retries, latency per agent,
and the latency of LLM routing vs deterministic routing. Output: results/trace_route_audit.{md,json}.
"""

import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results"


def fetch(dsn: str) -> tuple[list[dict], list[dict], list[dict]]:
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(dsn, connect_timeout=5, row_factory=dict_row) as conn:
        tops = conn.execute("SELECT request_id::text, task_type, status, routing_source, agents_called, duration_ms, "
                            "started_at FROM agent_runs WHERE parent_run_id IS NULL ORDER BY started_at").fetchall()
        kids = conn.execute("SELECT parent_run_id::text AS parent, agent_name, status, duration_ms, attempt, "
                            "tool_call_count FROM agent_runs WHERE parent_run_id IS NOT NULL").fetchall()
        msgs = conn.execute("SELECT r.request_id::text AS run, coalesce(r.parent_run_id, r.request_id)::text AS req, "
                            "m.message_type, m.recipient, m.content, m.duration_ms FROM agent_messages m "
                            "JOIN agent_runs r ON m.run_id = r.request_id").fetchall()
    return tops, kids, msgs


def audit(tops, kids, msgs) -> dict:
    from app.agents.routing import FALLBACK_ROUTES
    from app.agents.supervisor import SPECIALISTS

    children, skips, llm, tools = defaultdict(list), defaultdict(list), defaultdict(list), Counter()
    tool_retries = 0
    for k in kids:
        children[k["parent"]].append(k)
    for m in msgs:
        if m["message_type"] == "skip":
            skips[m["req"]].append((m["recipient"], (m["content"] or {}).get("reason", "")))
        elif m["message_type"] == "llm_call" and m["run"] == m["req"]:
            llm[m["req"]].append(m["duration_ms"])
        elif m["message_type"] == "tool_call":
            tools[m["req"]] += 1
            tool_retries += int(((m["content"] or {}).get("attempt") or 1) > 1)

    per_task = defaultdict(lambda: {"requests": 0, "statuses": Counter(), "executed": Counter(), "skipped": Counter(),
                                    "skip_reasons": Counter(), "never_run": set(), "route_accounted": 0,
                                    "ran_outside_route": 0, "agent_ms": defaultdict(list), "duration_ms": [],
                                    "routing": Counter()})
    traced = [t for t in tops if children.get(t["request_id"]) or skips.get(t["request_id"])]
    for t in traced:
        rid, task = t["request_id"], t["task_type"]
        p = per_task[task]
        p["requests"] += 1
        p["statuses"][t["status"]] += 1
        p["routing"][t["routing_source"]] += 1
        p["duration_ms"].append(t["duration_ms"])
        expected = set(FALLBACK_ROUTES.get(task, ()))
        executed = {k["agent_name"] for k in children[rid]}
        skipped = {a for a, _ in skips[rid]}
        # every agent the task needs either ran or was skipped with a recorded reason, and nothing else ran
        p["route_accounted"] += int(executed | skipped == expected)
        p["ran_outside_route"] += int(bool(executed - expected))
        for a in executed:
            p["executed"][a] += 1
        for a, reason in skips[rid]:
            p["skipped"][a] += 1
            p["skip_reasons"][f"{a}: {reason}"] += 1
        for k in children[rid]:
            p["agent_ms"][k["agent_name"]].append(k["duration_ms"])
        p["never_run"] |= set(SPECIALISTS) - set(expected)

    def med(v):
        return round(statistics.median(v), 1) if v else None

    tasks = {}
    for task, p in sorted(per_task.items()):
        never = sorted(p["never_run"])
        tasks[task] = {
            "requests": p["requests"], "statuses": dict(p["statuses"]), "routing": dict(p["routing"]),
            "expected_route": list(FALLBACK_ROUTES.get(task, ())), "route_accounted": p["route_accounted"],
            "requests_running_an_agent_outside_the_route": p["ran_outside_route"],
            "executed": dict(p["executed"]), "skipped": dict(p["skipped"]), "skip_reasons": dict(p["skip_reasons"]),
            "agents_not_needed": never,
            "agents_not_needed_that_ran": sorted(a for a in never if p["executed"].get(a)),
            "median_agent_ms": {a: med(v) for a, v in p["agent_ms"].items()},
            "median_request_ms": med(p["duration_ms"]),
        }
    llm_routed = [t["duration_ms"] for t in traced if t["routing_source"] == "llm"]
    det_routed = [t["duration_ms"] for t in traced if t["routing_source"] == "deterministic_fallback"]
    routing_llm_ms = [sum(v) for v in llm.values()]
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "requests": len(tops), "traced_requests": len(traced),
            "untraced_requests": len(tops) - len(traced),
            "tool_calls": sum(tools.values()), "tool_retries": tool_retries,
            "agent_retries": sum(1 for k in kids if (k["attempt"] or 1) > 1),
            "llm_routed_requests": len(llm_routed), "median_ms_llm_routed": med(llm_routed),
            "median_ms_deterministic": med(det_routed), "median_llm_routing_ms_per_request": med(routing_llm_ms),
            "tasks": tasks}


def render(r: dict) -> str:
    out = ["# Trace route audit (RQ9 evidence)", "",
           f"Generated by `src/evaluation/trace_route_audit.py` on {r['generated_at']} from the live `agent_runs` / "
           f"`agent_messages` rows. {r['requests']} persisted requests, {r['traced_requests']} with execution traces "
           f"(Phase 12 and later; {r['untraced_requests']} older rows have none). {r['tool_calls']} tool calls, "
           f"{r['tool_retries']} tool retries, {r['agent_retries']} agent retries.", "",
           "| Task | Requests | Expected route (FALLBACK_ROUTES) | Executed + skipped = route | Agents not needed for the task | ...that ran anyway | Executed | Skipped (reason) | Median request ms | Median per agent ms |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for task, t in r["tasks"].items():
        out.append(f"| `{task}` | {t['requests']} ({t['statuses']}) | {', '.join(t['expected_route']) or '—'} | "
                   f"{t['route_accounted']}/{t['requests']} | {', '.join(t['agents_not_needed']) or '—'} | "
                   f"{len(t['agents_not_needed_that_ran'])} | {t['executed']} | {t['skip_reasons'] or '—'} | "
                   f"{t['median_request_ms']} | {t['median_agent_ms']} |")
    out += ["", f"LLM routing: {r['llm_routed_requests']} traced request(s) were routed by the LLM, median "
            f"{r['median_ms_llm_routed']} ms per request, vs {r['median_ms_deterministic']} ms for deterministically "
            f"routed ones; the routing LLM calls themselves took a median {r['median_llm_routing_ms_per_request']} ms "
            "per request (including failed 429 calls before Phase 12's quota fix).", ""]
    return "\n".join(out)


if __name__ == "__main__":
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set")
    report = audit(*fetch(dsn))
    md = render(report)
    (OUT / "trace_route_audit.md").write_text(md, encoding="utf-8")
    (OUT / "trace_route_audit.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(md)
