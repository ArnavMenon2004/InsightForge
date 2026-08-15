"""
InsightForge LangGraph pipeline.

Defines the StateGraph that orchestrates all five agents in sequence.
Agents are kept entirely separate from the graph — the graph imports
from agents/, agents never import from graphs/.

Graph topology
--------------

    START
      │
      ▼
  [plan]          PlannerAgent     → ResearchPlan
      │
      ▼
  [research]      ResearcherAgent  → ResearchDocument
      │
      ▼
  [write]         WriterAgent      → FinalReport (draft)
      │
      ▼
  [fact_check]    FactCheckerAgent → FinalReport (+ fact_check)
      │
      ▼
  [cite]          CitationAgent    → FinalReport (+ citations)
      │
      ▼
    END

Each node reads from GraphState, calls its agent, and writes its
output back into GraphState.  If any node raises an exception it is
caught, the error field is set, and the graph routes to END via a
conditional edge so the UI always receives a usable state object.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.citation import CitationAgent
from agents.fact_checker import FactCheckerAgent
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from schemas.state import GraphState
from utils.logger import logger

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def plan_node(state: GraphState) -> dict:
    """Run PlannerAgent → populate state["plan"]."""
    logger.info("Graph: entering [plan] node.")
    try:
        result = PlannerAgent().run(
            query=state["query"],
            audience=state.get("audience"),
            depth=state.get("depth"),
        )
        return {"plan": result.data}
    except Exception as exc:
        logger.exception("Graph: [plan] node failed.")
        return {"error": str(exc)}


def research_node(state: GraphState) -> dict:
    """Run ResearcherAgent → populate state["document"]."""
    logger.info("Graph: entering [research] node.")
    try:
        result = ResearcherAgent().run(state["plan"])
        return {"document": result.data}
    except Exception as exc:
        logger.exception("Graph: [research] node failed.")
        return {"error": str(exc)}


def write_node(state: GraphState) -> dict:
    """Run WriterAgent → populate state["draft"]."""
    logger.info("Graph: entering [write] node.")
    try:
        result = WriterAgent().run(state["document"])
        return {"draft": result.data}
    except Exception as exc:
        logger.exception("Graph: [write] node failed.")
        return {"error": str(exc)}


def fact_check_node(state: GraphState) -> dict:
    """Run FactCheckerAgent → enrich state["draft"] with fact_check."""
    logger.info("Graph: entering [fact_check] node.")
    try:
        result = FactCheckerAgent().run(
            draft=state["draft"],
            document=state["document"],
        )
        return {"draft": result.data}
    except Exception as exc:
        logger.exception("Graph: [fact_check] node failed.")
        return {"error": str(exc)}


def cite_node(state: GraphState) -> dict:
    """Run CitationAgent → produce final state["report"]."""
    logger.info("Graph: entering [cite] node.")
    try:
        result = CitationAgent().run(
            draft=state["draft"],
            document=state["document"],
        )
        return {"report": result.data}
    except Exception as exc:
        logger.exception("Graph: [cite] node failed.")
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------


def _should_continue(state: GraphState) -> str:
    """Return 'continue' if no error, 'end' if a node failed."""
    return "end" if state.get("error") else "continue"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_pipeline() -> StateGraph:
    """
    Construct and compile the InsightForge StateGraph.

    Returns a compiled graph that accepts a ``GraphState`` dict and
    returns a completed ``GraphState`` dict.
    """
    graph = StateGraph(GraphState)

    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("write", write_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("cite", cite_node)

    graph.add_edge(START, "plan")

    for current, next_node in [
        ("plan",       "research"),
        ("research",   "write"),
        ("write",      "fact_check"),
        ("fact_check", "cite"),
    ]:
        graph.add_conditional_edges(
            current,
            _should_continue,
            {"continue": next_node, "end": END},
        )

    graph.add_edge("cite", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level compiled graph
# ---------------------------------------------------------------------------

pipeline = build_pipeline()


def get_graph_mermaid() -> str:
    """
    Return a Mermaid diagram string of the compiled pipeline graph.

    Used by the Streamlit sidebar to render a live architecture diagram.
    Returns an empty string if the diagram cannot be generated.
    """
    try:
        return pipeline.get_graph().draw_mermaid()
    except Exception:
        return ""