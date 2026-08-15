from __future__ import annotations

from agents.base_agent import BaseAgent
from prompts.planner import PLANNER_PROMPT
from schemas.research import ResearchPlan
from schemas.result import AgentResult


class PlannerAgent(BaseAgent):
    """
    Produces a ResearchPlan from a user query.

    Optionally accepts audience and depth overrides from the UI so the
    user's explicit selections take precedence over the LLM's inference.
    When overrides are provided they are injected into the query string
    so the planner prompt sees them as hard constraints.
    """

    prompt = PLANNER_PROMPT
    output_schema = ResearchPlan

    def __init__(self) -> None:
        super().__init__("Planner Agent")

    def run(
        self,
        query: str,
        audience: str | None = None,
        depth: str | None = None,
    ) -> AgentResult:
        """
        Generate a ResearchPlan for the given query.

        Args:
            query:    The user's research topic.
            audience: Optional audience override (e.g. "Developer").
                      When provided, appended to the query so the planner
                      treats it as a hard constraint.
            depth:    Optional depth override (e.g. "Advanced").

        Returns:
            AgentResult whose data is a ResearchPlan.
        """
        constrained_query = query

        if audience or depth:
            constraints: list[str] = []
            if audience:
                constraints.append(f"Target audience: {audience}")
            if depth:
                constraints.append(f"Research depth: {depth}")
            constrained_query = (
                f"{query}\n\n"
                f"[CONSTRAINTS — must be respected exactly]\n"
                + "\n".join(constraints)
            )

        return self.execute(query=constrained_query)