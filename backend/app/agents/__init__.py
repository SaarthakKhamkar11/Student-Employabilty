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
from app.embeddings.provider import EmbeddingProvider, build_embedding_provider, fallback_provider
from app.llm import LLMClient
from app.tools.explainability_tools import register_explainability_tools
from app.tools.job_tools import register_job_tools
from app.tools.learning_tools import register_learning_tools
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
                 matching_overrides: dict | None = None,
                 learning_repo: object = _UNSET, youtube_search=None,
                 fallback_embedder: EmbeddingProvider | None | object = _UNSET,
                 explain_model_loader=None) -> dict[str, BaseAgent]:
    """Instantiate all six agents and register their tools. Fails fast if any declared tool is missing.

    profile_repo / job_repo / matching_repo default to Postgres when DATABASE_URL is set; pass None for
    "no database" (the agents then report saved=false) or an in-memory repository in tests.
    matching_overrides: test hooks for the matching context (e.g. cfg, calibration_loader).
    explain_model_loader: test hook: (embedding_model, override) -> the model the Explainability Agent explains.
    """
    settings = settings or get_settings()
    registry = registry if registry is not None else ToolRegistry()
    embedder = embedder or build_embedding_provider(settings)
    if not registry.has("extract_text"):
        register_profile_tools(
            registry,
            repo=build_profile_repository(settings) if profile_repo is _UNSET else profile_repo,
            embedder=embedder,
            fallback_embedder=fallback_provider(embedder, settings) if fallback_embedder is _UNSET else fallback_embedder,
        )
    if not registry.has("extract_job_text"):
        register_job_tools(registry, repo=build_job_repository(settings) if job_repo is _UNSET else job_repo,
                           settings=settings)
    if matching_repo is _UNSET:
        matching_repo = build_matching_repository(settings)
    if not registry.has("filter_eligible"):
        register_matching_tools(registry, repo=matching_repo, embedder=embedder, settings=settings,
                                **(matching_overrides or {}))
    if not registry.has("prioritize_gaps"):
        from app.learning.repository import build_learning_repository
        from app.learning.youtube import search_videos

        register_learning_tools(
            registry, repo=build_learning_repository(settings) if learning_repo is _UNSET else learning_repo,
            embedder=embedder, settings=settings, youtube_search=youtube_search or search_videos)
    if not registry.has("get_shap_values"):
        # the LLM only restates validated evidence; without one, the structured evidence is returned
        register_explainability_tools(registry, repo=matching_repo, settings=settings, llm=llm,
                                      default_embedding_model=embedder.model_name,
                                      **({"model_loader": explain_model_loader} if explain_model_loader else {}))
    specialists = {cls.name: cls(registry=registry, settings=settings) for cls in SPECIALIST_CLASSES}
    supervisor = SupervisorAgent(agents=specialists, llm=llm, store=store, registry=registry, settings=settings)
    return {"supervisor": supervisor, **specialists}


__all__ = ["AGENT_CLASSES", "SPECIALIST_CLASSES", "AgentConfigurationError", "AgentContext", "BaseAgent",
           "SupervisorAgent", "build_agents"]
