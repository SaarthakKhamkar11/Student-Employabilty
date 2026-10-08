"""The six agents (spec §2). Agents reason, route and call tools; they are not ML models (spec §41)."""

from app.agents.base import AgentConfigurationError, AgentContext, BaseAgent
from app.agents.explainability_agent import ExplainabilityAgent
from app.agents.job_intelligence_agent import JobIntelligenceAgent
from app.agents.learning_agent import LearningAgent
from app.agents.matching_agent import MatchingAgent
from app.agents.profile_agent import ProfileAgent
from app.agents.supervisor import SupervisorAgent
from app.core.config import Settings
from app.db.store import AgentRunStore
from app.llm import LLMClient
from app.tools.registry import ToolRegistry

SPECIALIST_CLASSES: tuple[type[BaseAgent], ...] = (
    ProfileAgent, JobIntelligenceAgent, MatchingAgent, LearningAgent, ExplainabilityAgent,
)
AGENT_CLASSES: tuple[type[BaseAgent], ...] = (SupervisorAgent, *SPECIALIST_CLASSES)


def build_agents(registry: ToolRegistry | None = None, settings: Settings | None = None,
                 llm: LLMClient | None = None, store: AgentRunStore | None = None) -> dict[str, BaseAgent]:
    """Instantiate all six agents. Fails fast if any agent declares an unregistered tool."""
    specialists = {cls.name: cls(registry=registry, settings=settings) for cls in SPECIALIST_CLASSES}
    supervisor = SupervisorAgent(agents=specialists, llm=llm, store=store, registry=registry, settings=settings)
    return {"supervisor": supervisor, **specialists}


__all__ = ["AGENT_CLASSES", "SPECIALIST_CLASSES", "AgentConfigurationError", "AgentContext", "BaseAgent",
           "SupervisorAgent", "build_agents"]
