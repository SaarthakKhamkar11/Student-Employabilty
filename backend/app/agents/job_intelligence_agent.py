"""Job Intelligence Agent (spec §5.3), implemented in Phase 5.

Pipeline, with each stage a validated tool call through the BaseAgent loop:

    stored:  load_job_record → detect_pii → parse_job_description → extract_job_skills → normalize_skills
             → classify_skill_requirements → extract_requirements → check_duplicates → extract_job_posting
             → generate_embedding → validate_job_posting → save_job
    text:    extract_job_text → detect_pii → …(same)

detect_pii, normalize_skills and generate_embedding are the Profile Agent's
own tools, so students and jobs share one PII policy, one skill vocabulary
and one vector space. Required vs preferred skills are INFERRED
(docs/adr/0008). gender_preference only feeds the visibility filter
(docs/adr/0009), never a scoring field.

Duplicates: a stored job that is a near-duplicate is saved with
near_duplicate_of set. A new free-text JD that duplicates a stored job is
not saved again; its posting points at the existing job_id.
"""

import uuid
from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.core.sanitize import mask_pii
from app.schemas.agent_io import (
    AgentError, CallTool, EmbeddingInfo, Finish, JobIntelligenceInput, JobIntelligenceOutput,
)
from app.schemas.agent_state import INSUFFICIENT_EVIDENCE, NOT_YET_EVALUATED, SKILL_NOT_DETECTED
from app.tools.job_tools import RECORD_TEXT_FIELDS

COMMON = ("detect_pii", "parse_job_description", "extract_job_skills", "normalize_skills",
          "classify_skill_requirements", "extract_requirements", "check_duplicates", "extract_job_posting",
          "generate_embedding", "validate_job_posting", "save_job")
RECORD_PIPELINE = ("load_job_record", *COMMON)
TEXT_PIPELINE = ("extract_job_text", *COMMON)

# Stages that may fail without losing the posting, and which later stages each failure rules out.
DEGRADE_POLICY: dict[str, tuple[str, ...]] = {
    "detect_pii": ("generate_embedding", "save_job"),  # PII status unknown: nothing to ML or storage
    "extract_job_skills": ("normalize_skills", "classify_skill_requirements"),  # skills "Not yet evaluated"
    "normalize_skills": (),  # skills kept as written, marked "unmapped"
    "classify_skill_requirements": (),  # required/preferred "Not yet evaluated"
    "extract_requirements": (),  # degree and experience "Not yet evaluated"
    "check_duplicates": ("save_job",),  # unverified: never store (fail closed)
    "generate_embedding": (),  # saved without an embedding
    "save_job": (),  # returned unsaved
}


class JobIntelligenceAgent(BaseAgent):
    name = "job_intelligence_agent"
    role_description = (
        "You are the Job Intelligence Agent. Turn a job description into a structured job: title, role, "
        "responsibilities, required skills, preferred skills, education and experience requirements, work type "
        "and location. Keep required and preferred skills strictly separate and say that the split is inferred, "
        "normalise skills with the shared vocabulary, flag likely duplicate jobs, and request a job embedding "
        "from the currently selected embedding model through a tool. Never treat an employer's gender "
        "preference as a matching signal. "
        + GUARDRAILS
    )
    allowed_tools: ClassVar[list[str]] = sorted(set(RECORD_PIPELINE) | set(TEXT_PIPELINE))
    planned_tools = (  # spec §20
        "parse_job_description",
        "extract_requirements",
        "extract_skills",
        "normalize_skills",
        "generate_job_embedding",
        "save_job",
    )
    input_schema = JobIntelligenceInput
    output_schema = JobIntelligenceOutput
    implemented = True
    implemented_in_phase = 5

    def max_tool_retries(self) -> int:
        return 1

    def max_steps(self) -> int:
        return len(RECORD_PIPELINE) + 2

    # -- decisions -------------------------------------------------------------
    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        x = self._init(ctx)
        done = {o.tool_name for o in ctx.observations}
        for stage in x["pipeline"]:
            if stage in done or stage in x["stages"]:
                continue
            arguments = None if stage in x["skip"] else self._arguments(stage, ctx)
            if arguments is None:
                x["stages"][stage] = "skipped"
                continue
            return CallTool(tool_name=stage, arguments=arguments)
        return Finish(reason="job pipeline complete")

    def continue_after_failure(self, ctx: AgentContext, action: CallTool, error: AgentError) -> bool:
        ctx.extras["stages"][action.tool_name] = "failed"
        if action.tool_name in DEGRADE_POLICY:
            ctx.extras["skip"].update(DEGRADE_POLICY[action.tool_name])
            return True
        return False

    async def fallback(self, ctx: AgentContext, error: AgentError):
        output = self._output(ctx)
        if isinstance(output.posting, str):
            return "failed", INSUFFICIENT_EVIDENCE
        return "partial", output.model_dump(mode="json")

    async def finalize(self, ctx: AgentContext) -> JobIntelligenceOutput:
        output = self._output(ctx)
        if isinstance(output.embedding, EmbeddingInfo):
            ctx.state.model_versions.update(embedding_model=output.embedding.model_name,
                                            embedding_version=output.embedding.model_version)
        return output

    # -- helpers ---------------------------------------------------------------
    def _init(self, ctx: AgentContext) -> dict[str, Any]:
        x = ctx.extras
        if "pipeline" not in x:
            req: JobIntelligenceInput = ctx.input
            x["is_text"] = req.jd_text is not None
            x["pipeline"] = TEXT_PIPELINE if x["is_text"] else RECORD_PIPELINE
            x["job_id"] = req.job_id or f"JU{uuid.uuid4().hex[:10]}"  # uploaded JD: new id
            x["stages"], x["skip"] = {}, set()
        return x

    def _out(self, ctx: AgentContext, tool: str) -> BaseModel | None:
        for obs in reversed(ctx.observations):
            if obs.tool_name == tool:
                return obs.output
        return None

    def _record(self, ctx: AgentContext) -> dict[str, Any] | None:
        loaded = self._out(ctx, "load_job_record")
        return loaded.record if loaded is not None and loaded.found else None

    def _raw_fields(self, ctx: AgentContext) -> dict[str, str]:
        text = self._out(ctx, "extract_job_text")
        if text is not None:
            return {"jd": text.text}
        record = self._record(ctx) or {}
        return {f: str(record[f]) for f in RECORD_TEXT_FIELDS if record.get(f)}

    def _masked(self, ctx: AgentContext) -> dict[str, str]:
        pii = self._out(ctx, "detect_pii")
        if pii is not None:
            return pii.masked_fields
        return {k: mask_pii(v) for k, v in self._raw_fields(ctx).items()}  # conservative fallback

    def _masked_record(self, ctx: AgentContext) -> dict[str, Any] | None:
        record = self._record(ctx)
        return {**record, **self._masked(ctx)} if record else None

    def _skills_with_origin(self, ctx: AgentContext) -> list[dict[str, Any]] | str:
        extracted, normalized = self._out(ctx, "extract_job_skills"), self._out(ctx, "normalize_skills")
        if extracted is None:
            return NOT_YET_EVALUATED
        if not extracted.skills:
            return SKILL_NOT_DETECTED
        origin = {s.phrase: s.origin for s in extracted.skills}
        if normalized is not None:
            return [dict(s.model_dump(), origin=origin.get(s.raw)) for s in normalized.skills]
        return [{"raw": s.phrase, "canonical": s.phrase, "category": "unclassified", "method": "unmapped",
                 "evidence": None, "origin": s.origin} for s in extracted.skills]

    def _arguments(self, stage: str, ctx: AgentContext) -> dict[str, Any] | None:
        req: JobIntelligenceInput = ctx.input
        x = ctx.extras
        sections = self._out(ctx, "parse_job_description")
        if stage == "extract_job_text":
            return {"text": req.jd_text}
        if stage == "load_job_record":
            return {"job_id": req.job_id, "required": True}
        if stage == "detect_pii":
            fields = self._raw_fields(ctx)
            return {"fields": fields} if fields else None
        if stage == "parse_job_description":
            return {"text": self._masked(ctx)["jd"]} if x["is_text"] else {"record": self._masked_record(ctx)}
        if stage == "extract_job_skills":
            return {"sections": sections.sections.model_dump()}
        if stage == "normalize_skills":
            extracted = self._out(ctx, "extract_job_skills")
            return {"skills": [s.phrase for s in extracted.skills]} if extracted and extracted.skills else None
        if stage == "classify_skill_requirements":
            skills = self._skills_with_origin(ctx)
            if isinstance(skills, str):
                return None
            s = sections.sections
            return {"skills": skills, "description": s.description or "", "responsibilities": " ".join(s.responsibilities)}
        if stage == "extract_requirements":
            s = sections.sections
            return {"qualification_text": s.qualification_text, "experience_text": s.experience_text}
        if stage == "check_duplicates":
            s, record = sections.sections, self._record(ctx) or {}
            return {"job": {"job_id": req.job_id, "company_name": record.get("company_name"),
                            "job_title": record.get("job_title") or s.title,
                            "job_description": s.description, "responsibilities": " ".join(s.responsibilities),
                            "skills": s.skills_text or s.required_text}}
        if stage == "extract_job_posting":
            classified, requirements = self._out(ctx, "classify_skill_requirements"), self._out(ctx, "extract_requirements")
            duplicate, pii = self._out(ctx, "check_duplicates"), self._out(ctx, "detect_pii")
            skills = self._skills_with_origin(ctx)
            job_id = x["job_id"]
            if x["is_text"] and duplicate is not None and duplicate.near_duplicate_of:
                job_id = duplicate.near_duplicate_of  # the system already knows this job
            return {
                "job_id": job_id, "source": "job_description_text" if x["is_text"] else "stored_record",
                "record": {k: v for k, v in (self._masked_record(ctx) or {}).items() if k != "jd"} or None,
                "sections": sections.sections.model_dump(),
                "skills": [s.model_dump() for s in classified.skills] if classified else
                          (skills if isinstance(skills, str) else NOT_YET_EVALUATED),
                "split": classified.split.model_dump() if classified else NOT_YET_EVALUATED,
                "requirements": requirements.model_dump() if requirements else NOT_YET_EVALUATED,
                "duplicate_check": duplicate.model_dump() if duplicate else NOT_YET_EVALUATED,
                "pii": pii.report.model_dump() if pii else NOT_YET_EVALUATED,
            }
        posting = self._out(ctx, "extract_job_posting")
        if posting is None:
            return None
        if stage == "generate_embedding":
            return {"section_texts": job_section_texts(posting.model_dump())}
        if stage == "validate_job_posting":
            return {"posting": posting.model_dump(mode="json")}
        if stage == "save_job":
            validation, duplicate = self._out(ctx, "validate_job_posting"), self._out(ctx, "check_duplicates")
            if validation is None or not validation.valid:
                if validation is not None:
                    self.record_error(ctx, "posting_invalid", "; ".join(validation.problems), "validate_job_posting")
                return None
            if x["is_text"] and duplicate is not None and duplicate.near_duplicate_of:
                return None  # already stored as an existing job: do not create a duplicate row
            embedding = self._out(ctx, "generate_embedding")
            return {"posting": posting.model_dump(mode="json"), "is_new": x["is_text"],
                    "embedding": embedding.model_dump() if embedding else None}
        return None

    def _output(self, ctx: AgentContext) -> JobIntelligenceOutput:
        x = ctx.extras
        stages = {**{o.tool_name: "success" for o in ctx.observations}, **x.get("stages", {})}
        posting, embedding, saved = (self._out(ctx, t) for t in ("extract_job_posting", "generate_embedding", "save_job"))
        stored = saved is not None and saved.saved
        info = (EmbeddingInfo(model_name=embedding.model_name, model_version=embedding.model_version,
                              dimension=embedding.dimension, sections=sorted(embedding.vectors),
                              interim=embedding.interim)
                if embedding is not None and embedding.vectors and stored else NOT_YET_EVALUATED)
        return JobIntelligenceOutput(posting=posting.model_dump(mode="json") if posting else NOT_YET_EVALUATED,
                                     embedding=info, saved=stored, stages=stages)


def job_section_texts(posting: dict[str, Any]) -> dict[str, str]:
    """Per-section text for embedding. Sections holding a sentinel are omitted, never embedded as empty."""
    def text(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(v["canonical"] if isinstance(v, dict) else str(v) for v in value)
        return "" if value in ("Not detected", "Not yet evaluated", "Skill not detected") else str(value)

    sections = {
        "skills": text(posting["skills"]),
        "required_skills": text(posting["required_skills"]),
        "responsibilities": text(posting["responsibilities"]),
        "description": text(posting["job_description"]),
        "education": text(posting["qualification_required"]),
    }
    sections = {k: v for k, v in sections.items() if v.strip()}
    if sections:
        sections["full"] = "\n".join(v for k, v in sections.items() if k != "required_skills")
    return sections
