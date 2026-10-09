"""Shared workflow state passed between the Supervisor and the agents (spec §5.1)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The only two values allowed for a result that does not exist (spec rule 19).
NOT_YET_EVALUATED = "Not yet evaluated"
INSUFFICIENT_EVIDENCE = "Insufficient evidence"
Unavailable = Literal["Not yet evaluated", "Insufficient evidence"]

# Extraction outcomes (spec §5.2): absence of evidence is not evidence of absence.
SKILL_NOT_DETECTED = "Skill not detected"  # the skills were looked for and none were found
NOT_DETECTED = "Not detected"  # a field was looked for and not found
# NOT_YET_EVALUATED also covers a field no extractor looks for yet, or a stage that failed.

WorkflowStatus = Literal["pending", "running", "success", "partial", "failed"]


class ModelVersionInfo(BaseModel):
    """Version metadata every prediction must carry (spec §24).

    Until Phases 6 and 8 select models from experimental evidence, every field
    is explicitly "Not yet evaluated", never a guess.
    """

    model_config = ConfigDict(extra="forbid")

    embedding_model: str = NOT_YET_EVALUATED
    embedding_version: str = NOT_YET_EVALUATED
    matching_model: str = NOT_YET_EVALUATED
    matching_model_version: str = NOT_YET_EVALUATED
    feature_version: str = NOT_YET_EVALUATED
    dataset_version: str = NOT_YET_EVALUATED
    experiment_id: str = NOT_YET_EVALUATED


class PredictionVersions(BaseModel):
    """The spec §24 metadata on every prediction, strictly: all seven fields are required and must be real
    values. A missing, empty or sentinel field fails validation, so such a prediction is never returned."""

    model_config = ConfigDict(extra="forbid")

    embedding_model: str
    embedding_version: str
    matching_model: str
    matching_model_version: str
    feature_version: str
    dataset_version: str
    experiment_id: str

    @field_validator("*")
    @classmethod
    def _real_value(cls, value: str) -> str:
        if not value or not value.strip() or value in (NOT_YET_EVALUATED, INSUFFICIENT_EVIDENCE):
            raise ValueError("prediction metadata must be a real value, not missing or a sentinel")
        return value


class AgentState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    request_id: str
    user_id: str | None = None
    task_type: str
    student_id: str | None = None
    job_id: str | None = None
    agents_called: list[str] = Field(default_factory=list)
    agent_results: dict[str, Any] = Field(default_factory=dict)
    model_versions: dict[str, str] = Field(default_factory=lambda: ModelVersionInfo().model_dump())
    errors: list[dict] = Field(default_factory=list)
    status: WorkflowStatus = "pending"
