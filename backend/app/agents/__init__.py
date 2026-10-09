"""The six agents (spec §2). Agents reason, route and call tools; they are not ML models (spec §41)."""

from app.agents.base import AgentConfigurationError, AgentContext, BaseAgent
from app.agents.explainability_agent import ExplainabilityAgent
from app.agents.job_intelligence_agent import JobIntelligenceAgent
from app.agents.learning_agent import LearningAgent
from app.agents.matching_agent import MatchingAgent
from app.agents.profile_agent import ProfileAgent
from app.agents.supervisor import SupervisorAgent
from app.core.config import Settings, get_settings
from app.db.job_repository import JobRepository, build_job_repository
from app.db.matching_repository import MatchingRepository, build_matching_repository
from app.db.profile_repository import ProfileRepository, build_profile_repository
from app.db.store import AgentRunStore
from app.embeddings.provider import EmbeddingProvider, build_embedding_provider
from app.llm import LLMClient
from app.tools.job_tools import register_job_tools
from app.tools.matching_tools import register_matching_tools
from app.tools.profile_tools import register_profile_tools
from app.tools.registry import ToolRegistry

_UNSET = object()

SPECIALIST_CLASSES: tuple[type[BaseAgent], ...] = (
    ProfileAgent, JobIntelligenceAgent, MatchingAgent, LearningAgent, ExplainabilityAgent,
)
AGENT_CLASSES: tuple[type[BaseAgent], ...] = (SupervisorAgent, *SPECIALIST_CLASSES)


def build_agents(registry: ToolRegistry | None = None, settings: Settings | None = None,
                 llm: LLMClient | None = None, store: AgentRunStore | None = None,
                 profile_repo: ProfileRepository | None | object = _UNSET,
                 job_repo: JobRepository | None | object = _UNSET,
                 embedder: EmbeddingProvider | None = None,
                 matching_repo: MatchingRepository | None | object = _UNSET,
                 matching_overrides: dict | None = None) -> dict[str, BaseAgent]:
    """Instantiate all six agents and register their tools. Fails fast if any declared tool is missing.

    profile_repo / job_repo / matching_repo default to Postgres when DATABASE_URL is set; pass None for
    "no database" (the agents then report saved=false) or an in-memory repository in tests.
    matching_overrides: test hooks for the matching context (e.g. cfg, calibration_loader).
    """
    settings = settings or get_settings()
    registry = registry if registry is not None else ToolRegistry()
    embedder = embedder or build_embedding_provider(settings)
    if not registry.has("extract_text"):
        register_profile_tools(
            registry,
            repo=build_profile_repository(settings) if profile_repo is _UNSET else profile_repo,
            embedder=embedder,
        )
    if not registry.has("extract_job_text"):
        register_job_tools(registry, repo=build_job_repository(settings) if job_repo is _UNSET else job_repo,
                           settings=settings)
    if not registry.has("filter_eligible"):
        register_matching_tools(
            registry, repo=build_matching_repository(settings) if matching_repo is _UNSET else matching_repo,
            embedder=embedder, settings=settings, **(matching_overrides or {}))
    specialists = {cls.name: cls(registry=registry, settings=settings) for cls in SPECIALIST_CLASSES}
    supervisor = SupervisorAgent(agents=specialists, llm=llm, store=store, registry=registry, settings=settings)
    return {"supervisor": supervisor, **specialists}


__all__ = ["AGENT_CLASSES", "SPECIALIST_CLASSES", "AgentConfigurationError", "AgentContext", "BaseAgent",
           "SupervisorAgent", "build_agents"]
