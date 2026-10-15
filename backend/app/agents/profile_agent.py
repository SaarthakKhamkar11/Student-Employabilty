"""Profile Agent (spec §5.2), implemented in Phase 4.

Pipeline, with each stage a validated tool call through the BaseAgent loop:

    resume:  extract_text → detect_pii → parse_resume → extract_skills → normalize_skills
             → extract_profile → generate_embedding → validate_profile → save_profile
    stored:  load_student_record → detect_pii → parse_student_sections → extract_skills → …(same)
    task cohort_benchmark (Phase 12): … → save_profile → cohort_benchmark (the stored profile against its peers)

When both a student_id and a resume are given, the stored record supplies the
fields a resume does not state (branch, CGPA, graduation year ...), and the
resume supplies the sections and skills.

Every stage after detect_pii works on PII-masked text. Degradation follows
DEGRADE_POLICY: a degradable stage that fails all retries is flagged, the
stages that depend on it are skipped, and the result is "partial". Any other
failure stops the run.
"""

import uuid
from typing import Any, ClassVar

from pydantic import BaseModel

from app.agents.base import GUARDRAILS, AgentContext, BaseAgent
from app.core.sanitize import mask_pii
from app.profile.sections import record_skills_text
from app.schemas.agent_io import (
    AgentError, CallTool, EmbeddingInfo, Finish, ProfileAgentInput, ProfileAgentOutput,
)
from app.schemas.agent_state import INSUFFICIENT_EVIDENCE, NOT_YET_EVALUATED, SKILL_NOT_DETECTED
from app.schemas.student import NormalizedSkill
from app.tools.profile_tools import RECORD_TEXT_FIELDS

RESUME_PIPELINE = ("extract_text", "detect_pii", "parse_resume", "extract_skills", "normalize_skills",
                   "extract_profile", "generate_embedding", "validate_profile", "save_profile")
RECORD_PIPELINE = ("load_student_record", "detect_pii", "parse_student_sections", "extract_skills",
                   "normalize_skills", "extract_profile", "generate_embedding", "validate_profile", "save_profile")

# Stages that may fail without losing the profile, and which later stages each failure rules out.
DEGRADE_POLICY: dict[str, tuple[str, ...]] = {
    # PII status unknown: nothing goes to the ML pipeline or to storage. Parsing continues on text
    # masked by the conservative email/phone mask (core.sanitize.mask_pii) instead.
    "detect_pii": ("generate_embedding", "save_profile"),
    # No skills extracted: skills read "Not yet evaluated", not "Skill not detected".
    "extract_skills": ("normalize_skills",),
    # Skills kept exactly as written and marked method="unmapped".
    "normalize_skills": (),
    # Profile saved without an embedding; embedding reads "Not yet evaluated".
    "generate_embedding": (),
    # Profile returned but not stored (saved=false).
    "save_profile": (),
    # Profile returned without the cohort comparison (cohort_benchmark reads "Insufficient evidence").
    "cohort_benchmark": (),
}


class ProfileAgent(BaseAgent):
    name = "profile_agent"
    role_description = (
        "You are the Profile Agent. Turn a student's resume or profile into a structured representation: "
        "education, technical and soft skills, projects, internships and experience, certifications and "
        "career interests. Detect PII so it can be kept out of the ML pipeline, normalise skills and their "
        "aliases, and request an embedding from the currently selected embedding model through a tool. A "
        "skill you cannot find is 'not detected', never 'not possessed'. "
        + GUARDRAILS
    )
    allowed_tools: ClassVar[list[str]] = sorted(set(RESUME_PIPELINE) | set(RECORD_PIPELINE) | {"cohort_benchmark"})
    planned_tools = (  # spec §20
        "parse_resume",
        "extract_profile",
        "extract_skills",
        "normalize_skills",
        "generate_embedding",
        "save_profile",
    )
    input_schema = ProfileAgentInput
    output_schema = ProfileAgentOutput
    implemented = True
    implemented_in_phase = 4

    def trace_model_versions(self, data: dict) -> dict[str, str]:
        e = data.get("embedding")
        return {"embedding": f"{e['model_name']} {e['model_version']}"} if isinstance(e, dict) else {}

    def max_tool_retries(self) -> int:
        return 1  # Phase 4 rule: a failing stage is retried once, then degraded or stopped

    def max_steps(self) -> int:
        # the longest path (stored record + resume, then cohort_benchmark) is 11 tool calls, plus the finish decision
        return len(RESUME_PIPELINE) + 3

    # -- decisions -------------------------------------------------------------
    async def decide_next_action(self, ctx: AgentContext) -> CallTool | Finish:
        x = self._init(ctx)
        loaded = self._out(ctx, "load_student_record")
        if not x["has_resume"] and loaded is not None and loaded.stored_profile is not None:
            # A resume upload asked for again by id: its raw resume is not stored, so rebuilding from the empty
            # stored columns would overwrite the real profile with an empty one. Reuse the stored profile instead.
            x["reused"] = loaded.stored_profile
            if self._wants_cohort(ctx) and "cohort_benchmark" not in x["stages"] \
                    and self._out(ctx, "cohort_benchmark") is None:
                return CallTool(tool_name="cohort_benchmark",
                                arguments={"student_id": x["student_id"], "job_id": ctx.state.job_id})
            return Finish(reason="reused the stored profile of a resume upload")
        done = {o.tool_name for o in ctx.observations}
        for stage in x["pipeline"]:
            if stage in done or stage in x["stages"]:
                continue
            if stage in x["skip"]:
                x["stages"][stage] = "skipped"
                continue
            arguments = self._arguments(stage, ctx)
            if arguments is None:
                x["stages"][stage] = "skipped"
                continue
            return CallTool(tool_name=stage, arguments=arguments)
        return Finish(reason="profile pipeline complete")

    def continue_after_failure(self, ctx: AgentContext, action: CallTool, error: AgentError) -> bool:
        x = ctx.extras
        x["stages"][action.tool_name] = "failed"
        if action.tool_name == "load_student_record":
            return x["has_resume"]  # the resume alone can still produce a profile
        if action.tool_name in DEGRADE_POLICY:
            x["skip"].update(DEGRADE_POLICY[action.tool_name])
            return True
        return False

    async def fallback(self, ctx: AgentContext, error: AgentError):
        output = self._output(ctx)
        if isinstance(output.profile, str):  # nothing usable was built
            return "failed", INSUFFICIENT_EVIDENCE
        return "partial", output.model_dump(mode="json")

    async def finalize(self, ctx: AgentContext) -> ProfileAgentOutput:
        output = self._output(ctx)
        if isinstance(output.embedding, EmbeddingInfo):  # spec §24: record which model produced this state
            ctx.state.model_versions.update(embedding_model=output.embedding.model_name,
                                            embedding_version=output.embedding.model_version)
        return output

    # -- helpers ---------------------------------------------------------------
    def _init(self, ctx: AgentContext) -> dict[str, Any]:
        x = ctx.extras
        if "pipeline" not in x:
            req: ProfileAgentInput = ctx.input
            x["has_resume"] = req.resume_text is not None or bool(req.resume_file_b64)
            x["pipeline"] = (("load_student_record",) if req.student_id else ()) + RESUME_PIPELINE \
                if x["has_resume"] else RECORD_PIPELINE
            x["student_id"] = req.student_id or f"U{uuid.uuid4().hex[:12]}"  # new upload: new id, never reused
            if self._wants_cohort(ctx):
                x["pipeline"] += ("cohort_benchmark",)
            x["stages"], x["skip"] = {}, set()
        return x

    @staticmethod
    def _wants_cohort(ctx: AgentContext) -> bool:
        return ctx.state.task_type == "cohort_benchmark"

    def _out(self, ctx: AgentContext, tool: str) -> BaseModel | None:
        for obs in reversed(ctx.observations):
            if obs.tool_name == tool:
                return obs.output
        return None

    def _record(self, ctx: AgentContext) -> dict[str, Any] | None:
        loaded = self._out(ctx, "load_student_record")
        return loaded.record if loaded is not None and loaded.found else None

    def _masked(self, ctx: AgentContext) -> dict[str, str]:
        """PII-masked inputs. If detect_pii failed, fall back to the conservative email/phone mask."""
        pii = self._out(ctx, "detect_pii")
        if pii is not None:
            return pii.masked_fields
        raw = self._raw_fields(ctx)
        return {k: mask_pii(v) for k, v in raw.items()}

    def _raw_fields(self, ctx: AgentContext) -> dict[str, str]:
        """Everything that will reach the profile: the resume text and/or the stored record's free text."""
        record = self._record(ctx) or {}
        fields = {f: str(record[f]) for f in RECORD_TEXT_FIELDS if record.get(f) not in (None, "")}
        text = self._out(ctx, "extract_text")
        if text is not None:
            fields["resume"] = text.text
        return fields

    def _masked_record(self, ctx: AgentContext) -> dict[str, Any]:
        masked = {k: v for k, v in self._masked(ctx).items() if k != "resume"}
        return {**(self._record(ctx) or {}), **masked}

    def _skills(self, ctx: AgentContext) -> list[NormalizedSkill] | str:
        stages = ctx.extras["stages"]
        normalized, extracted = self._out(ctx, "normalize_skills"), self._out(ctx, "extract_skills")
        if normalized is not None:
            return normalized.skills or SKILL_NOT_DETECTED
        if extracted is None:
            return NOT_YET_EVALUATED  # extraction failed or never ran: not the same as "not detected"
        if not extracted.skills:
            return SKILL_NOT_DETECTED
        if stages.get("normalize_skills") == "failed":  # keep them, honestly marked as not normalised
            return [NormalizedSkill(raw=s, canonical=s, category="unclassified", method="unmapped")
                    for s in extracted.skills]
        return NOT_YET_EVALUATED

    def _profile(self, ctx: AgentContext) -> dict[str, Any] | None:
        built = self._out(ctx, "extract_profile")
        return built.model_dump(mode="json") if built is not None else None

    def _arguments(self, stage: str, ctx: AgentContext) -> dict[str, Any] | None:
        """Arguments for a stage, built from earlier outputs, or None if the stage has nothing to do."""
        req: ProfileAgentInput = ctx.input
        x = ctx.extras
        if stage == "extract_text":
            return {"text": req.resume_text, "content_b64": req.resume_file_b64, "filename": req.resume_filename}
        if stage == "load_student_record":
            return {"student_id": req.student_id, "required": not x["has_resume"]}
        if stage == "detect_pii":
            fields = self._raw_fields(ctx)
            return {"fields": fields} if fields else None
        if stage == "parse_resume":
            return {"text": self._masked(ctx)["resume"]}
        if stage == "parse_student_sections":
            return {"record": self._masked_record(ctx)}
        if stage == "extract_skills":
            masked = self._masked(ctx)
            if "resume" in masked:
                return {"text": masked["resume"], "source": "resume"}
            return {"text": record_skills_text(self._masked_record(ctx)), "source": "record"}
        if stage == "normalize_skills":
            extracted = self._out(ctx, "extract_skills")
            return {"skills": extracted.skills} if extracted is not None and extracted.skills else None
        if stage == "extract_profile":
            sections = self._out(ctx, "parse_resume") or self._out(ctx, "parse_student_sections")
            pii = self._out(ctx, "detect_pii")
            record = self._masked_record(ctx) if self._record(ctx) else None
            source = ("stored_record+resume" if x["has_resume"] and record
                      else "resume" if x["has_resume"] else "stored_record")
            skills = self._skills(ctx)
            return {"student_id": x["student_id"], "source": source, "sections": sections.sections,
                    "skills": [s.model_dump() for s in skills] if isinstance(skills, list) else skills,
                    # gender is a protected attribute: never part of the profile (spec §11)
                    "record": {k: v for k, v in record.items() if k not in ("gender", "resume")} if record else None,
                    "pii": pii.report.model_dump() if pii is not None else NOT_YET_EVALUATED}
        if stage == "generate_embedding":
            profile = self._profile(ctx)
            return {"section_texts": section_texts(profile), "role": "query"} if profile else None
        if stage == "validate_profile":
            profile = self._profile(ctx)
            return {"profile": profile} if profile else None
        if stage == "save_profile":
            validation = self._out(ctx, "validate_profile")
            if validation is None or not validation.valid:
                if validation is not None:
                    self.record_error(ctx, "profile_invalid", "; ".join(validation.problems), "validate_profile")
                return None  # never store an unvalidated or invalid profile
            embedding = self._out(ctx, "generate_embedding")
            return {"profile": self._profile(ctx),
                    "embedding": embedding.model_dump() if embedding is not None else None}
        if stage == "cohort_benchmark":  # only a stored profile can be compared with its stored peers
            saved = self._out(ctx, "save_profile")
            if saved is None or not saved.saved:
                return None
            return {"student_id": x["student_id"], "job_id": ctx.state.job_id}
        return None

    def _cohort(self, ctx: AgentContext):
        found = self._out(ctx, "cohort_benchmark")
        if found is not None:
            return found
        return INSUFFICIENT_EVIDENCE if self._wants_cohort(ctx) else NOT_YET_EVALUATED

    def _output(self, ctx: AgentContext) -> ProfileAgentOutput:
        x = ctx.extras
        stages = {**{o.tool_name: "success" for o in ctx.observations}, **x.get("stages", {})}
        if x.get("reused") is not None:
            skipped = {s: "skipped" for s in RECORD_PIPELINE if s != "load_student_record"}
            return ProfileAgentOutput(profile=x["reused"], embedding=NOT_YET_EVALUATED, saved=False,
                                      stages={**skipped, **stages}, cohort_benchmark=self._cohort(ctx))
        embedding, saved = self._out(ctx, "generate_embedding"), self._out(ctx, "save_profile")
        stored = saved is not None and saved.saved
        info = (EmbeddingInfo(model_name=embedding.model_name, model_version=embedding.model_version,
                              dimension=embedding.dimension, sections=sorted(embedding.vectors),
                              interim=embedding.interim)
                if embedding is not None and embedding.vectors and stored else NOT_YET_EVALUATED)
        return ProfileAgentOutput(profile=self._profile(ctx) or NOT_YET_EVALUATED, embedding=info, saved=stored,
                                  stages=stages, cohort_benchmark=self._cohort(ctx))


def section_texts(profile: dict[str, Any]) -> dict[str, str]:
    """Text per section for embedding. Sections holding a sentinel are omitted, never embedded as empty."""
    def text(value: Any) -> str:
        if isinstance(value, list):
            return "\n".join(v["canonical"] if isinstance(v, dict) else str(v) for v in value)
        return "" if value in ("Not detected", "Not yet evaluated", "Skill not detected") else str(value)

    sections = {
        "skills": text(profile["skills"]),
        "experience": text(profile["experience"]),
        "education": text(profile["education"]),
        "projects": text(profile["projects"]),
        "certifications": text(profile["certifications"]),
    }
    sections = {k: v for k, v in sections.items() if v.strip()}
    if sections:
        sections["full"] = "\n".join(sections.values())
    return sections
