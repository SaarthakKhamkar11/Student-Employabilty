# ADR 0020: Agent tracing from real spans, request-scoped PII redaction, and aggregate-only cohort benchmarking

**Status:** Accepted, 2026-10-15 (Phase 12).

**Context.**
- **Spec §39:** trace every agent execution: `request_id, agent_name, task, start_time, end_time, duration, tool_calls,
  tool_latency, model_used, model_version, status, error, retry_count`. Never log sensitive PII. The research
  dashboard (Phase 13) shows the Supervisor → agents tree, to show the system really is agentic.
- **Spec §18:** benchmark a student against a cohort filtered by branch, target role and graduation year, then narrowed
  with embeddings. Report the cohort average, percentile, skill coverage and relative skill gap. Never expose another
  student's private information.
- **Already there:** `agent_runs` (one row per request, Phase 3); ADR 0017's tree columns (`parent_run_id`,
  `agent_name`, `retry_count`, `used_fallback`) and the `agent_messages` table, neither written yet; JSON logs with
  key- and pattern-based redaction; the Phase 4 PII detector (`app/profile/pii.py`), which masks only the spans it
  finds.

**Decision.**

1. **Spans are recorded by the runtime, at the moment things run** (`app/core/tracing.py`). The Supervisor opens one
   `TraceRecorder` per request. A context variable carries the current span, so parallel agent waves
   (`asyncio.gather`) and synchronous tools (`asyncio.to_thread`) nest correctly.

   | Span | Recorded by | One per |
   |---|---|---|
   | agent | `BaseAgent.run` | execution. A Supervisor retry is a second span, attempt 2 |
   | tool | `BaseAgent._call_tool_with_retry` | attempt, so tool retries are visible |
   | llm | `TracedLLMClient`, wrapped in the `Router` and the explainability tools | provider call: model, purpose, latency, HTTP status (429 = quota) |
   | skip | `SupervisorAgent.delegate` | planned agent not run, with the reason (zero duration) |

   - Each agent reports its own `model_version` from its output (`trace_model_versions`). For example, matching
     reports `lightgbm+<hash>`, and profile reports `e5_base_v2 <revision>`.
2. **Persisted in the spec's tables, not only in logs** (migration `0011_agent_tracing.sql`).
   - The Supervisor's `agent_runs` row is the request.
   - Each agent execution gets its own `agent_runs` row (its own UUID, `parent_run_id` = the request). New columns:
     `attempt`, `termination_reason`, `tool_call_count`, `tool_latency_ms`, `model_version`, `error_code`.
   - Each tool attempt, LLM call and skip is an `agent_messages` row of the execution it happened in. The new
     `span_id` / `parent_span_id` columns nest an LLM call inside its tool attempt.
   - All rows are written in one transaction with the request row.
3. **The tree is rebuilt only from those rows** (`app/core/trace_tree.py`). `get_agent_trace(request_id)` returns a
   `TraceTree`; `GET /requests/{id}/trace` returns it plus a text rendering.
   - A request that was never persisted has no trace (`None` / 404), never an example one.
   - Tests check the tree against each agent's own `AgentOutput.tool_calls`, and the Postgres tree against the
     in-memory one.
4. **Request-scoped PII redaction, fed by the Phase 4 detector** (`app/core/logging.py`).
   - `detect_pii` registers every value it flags in a per-request in-memory set, as does the Supervisor's own scan of
     `resume_text` before anything runs. Names are registered word by word, phone numbers also as bare digits.
   - From then on, `redact_text` replaces those values in every log line and every persisted trace record, wherever
     they appear, on top of the existing key and pattern rules.
   - Why: the detector masks the name only where it finds it (the first line). A copy elsewhere in the resume flows
     into the profile, and it used to reach `agent_runs.agent_results`.
   - The registered values are never logged or stored, and they die with the request.
5. **Cohort benchmarking** (`src/scoring/cohort_analytics.py`, `config/cohort.yaml`; the `cohort_benchmark` tool).
   - **Stage 1, filters.** Branch, graduation year and target role, read from the stored profile.
     - The target role is the job's role family when a job is given. Otherwise it is the student's own: their
       career interest's family, or their domain's primary family when the interest is a plan such as
       "Placements". Families and the domain map come from `config/weak_label_rules.yaml` (Phases 6–7).
     - A filter applies only when the student's own value is known. Kaggle students have no graduation year, so
       that filter isn't applied to them, rather than guessed.
     - Fewer than 5 peers left: graduation year, then target role, are dropped, and this is reported. Branch never is.
   - **Stage 2, comparable students.** The top 20 peers by cosine similarity of the stored **profile-level** (`full`)
     embedding of the configured model (E5). Profile-level rather than skills-only, because 35 of the 286 students
     have no skills section.
   - **Stage 3, statistics,** over those peers.
     - **Suitability:** each peer and the student scored for the same job with the Matching Agent's own Stages 2–4.
       Hard eligibility rules are not applied (the comparison is of suitability, not of who may apply). Reported:
       cohort average, median, and the student's percentile (ties count half). A test checks that the student's own
       score equals the Matching Agent's for the same pair.
     - **Skill coverage:** the share of the cohort's 10 most common skills the student has.
     - **Relative skill gap:** each missing job skill against the share of peers who have it. "Distinctive" gaps
       (most peers have the skill) are the priorities; "shared" gaps (most peers lack it too) and "advantages" (the
       student has it, most peers don't) are listed separately.
   - **Callers.** The Matching Agent runs it for the top-ranked job in student modes, as its last stage (a failure
     leaves the matches and makes the run partial). The Profile Agent runs it for task `cohort_benchmark`, which the
     router already sent to it alone.
6. **Privacy: aggregates only.** The response carries counts, averages, medians, percentiles and shares of peers,
   never a peer id, score or profile.
   - Every statistic needs at least 5 peers.
   - A common skill is listed only if at least 3 peers hold it.
   - A test checks responses for students in two different branches against all 285 other real `student_id`s.

**Consequences.**
- Phase 13 can query execution history directly: per-agent latency, tool latency, retries, LLM calls and 429s per
  agent.
- The live trace already shows something the logs never did: the Learning Agent's `compute_course_similarity` takes
  13–15 s of a ~20 s resume upload.
- **The PII registry protects logs and traces, not the data.** A name repeated in a resume body still reaches the
  stored profile (`student_profiles`) and its embeddings, because Phase 4 masks only the spans it finds. Making
  `mask()` replace every occurrence of each detected value is recommended for Phase 16.
- With LightGBM, scores for one strong job cluster near 1 (cohort averages around 0.999), so a percentile there can
  separate students whose scores differ only in the 4th decimal. The response gives the raw averages alongside it,
  so this stays visible.
- An uploaded resume without a detected branch or graduation year is benchmarked against the target-role filter
  alone, and the response says so. LightGBM's top feature is `branch_match`, so such a profile scores near 0
  against peers who have one.
- The matching path costs one more tool call per student request (0.8–1.6 s live on E5).
