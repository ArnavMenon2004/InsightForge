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
  [contradict]    ContradictionAgent → FinalReport (+ contradiction_report)
      │
      ▼
  [diversity]     rule-based       → FinalReport (+ source_diversity)
      │
      ▼
    END

Each node reads from GraphState, calls its agent, and writes its
output back into GraphState.  If any node raises an exception it is
caught, the error field is set, and the graph routes to END via a
conditional edge so the UI always receives a usable state object.
"""

from __future__ import annotations

from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph

from agents.citation import CitationAgent
from agents.contradiction import ContradictionAgent
from agents.fact_checker import FactCheckerAgent
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.writer import WriterAgent
from schemas.report import SourceDiversityReport
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


def contradict_node(state: GraphState) -> dict:
    """
    Run ContradictionAgent → enrich state["report"] with contradiction_report.

    The report is already complete at this point, so a failure here is
    logged and skipped rather than discarding the finished report.
    """
    logger.info("Graph: entering [contradict] node.")
    try:
        result = ContradictionAgent().run(
            report=state["report"],
            document=state["document"],
        )
        return {"report": result.data}
    except Exception:
        logger.exception("Graph: [contradict] node failed; skipping.")
        return {}


_ACADEMIC = (
    "arxiv.org", "ieee.org", "acm.org", "nature.com", "science.org", "sciencedirect.com",
    "springer.com", "springeropen.com", "wiley.com", "tandfonline.com", "sagepub.com",
    "aps.org", "aip.org", "iop.org", "frontiersin.org", "mdpi.com", "plos.org", "cell.com",
    "preprints.org", "biorxiv.org", "medrxiv.org", "ssrn.com", "openreview.net", "doi.org",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "researchgate.net", "semanticscholar.org", "jstor.org",
)
_NEWS = (
    "reuters.com", "bbc.com", "bbc.co.uk", "cnn.com", "nytimes.com", "theguardian.com",
    "forbes.com", "bloomberg.com", "wsj.com", "ft.com", "economist.com", "cnbc.com",
    "techcrunch.com", "wired.com", "theverge.com", "arstechnica.com", "zdnet.com",
    "venturebeat.com", "technologyreview.com", "quantamagazine.org", "newscientist.com",
    "phys.org", "sciencedaily.com", "sciencemediacentre.org", "prnewswire.com",
    "businesswire.com", "globenewswire.com", "quantumcomputingreport.com", "thequantuminsider.com",
)
_INDUSTRY_HOSTS = (
    "ibm.com", "microsoft.com", "aws.amazon.com", "google.com", "nvidia.com",
    "quantinuum.com", "ionq.com", "quera.com", "rigetti.com", "dwavesys.com",
)
_INDUSTRY_PATHS = ("/blog", "/insights", "/press", "/whitepaper", "/resources")


def _classify_source(url: str) -> str:
    """Rule-based source-type classifier (no LLM), matched on the URL's host."""
    parsed = urlparse(url.lower())
    host = parsed.hostname or ""

    def on(domains) -> bool:
        return any(host == d or host.endswith("." + d) for d in domains)

    if on(_ACADEMIC):
        return "academic"
    if on(_NEWS):
        return "news"
    if (
        host.endswith((".gov", ".edu", ".mil", ".gov.uk", ".ac.uk", ".gc.ca"))
        or on(("who.int", "un.org", "europa.eu", "oecd.org", "worldbank.org"))
    ):
        return "government"
    if on(_INDUSTRY_HOSTS) or any(p in parsed.path for p in _INDUSTRY_PATHS) or "/blog" in host:
        return "industry"
    return "other"


def diversity_node(state: GraphState) -> dict:
    """Classify all sources by domain type → state["report"].source_diversity."""
    logger.info("Graph: entering [diversity] node.")
    try:
        counts = {"academic": 0, "news": 0, "industry": 0, "government": 0, "other": 0}
        seen: set[str] = set()
        for section in state["document"].sections:
            for source in section.sources:
                url = str(source.url)
                if url in seen:
                    continue
                seen.add(url)
                counts[_classify_source(url)] += 1

        types_present = sum(1 for n in counts.values() if n > 0)
        diversity = SourceDiversityReport(
            **counts,
            diversity_score="High" if types_present >= 3 else "Medium" if types_present == 2 else "Low",
            dominant_type=max(counts, key=counts.get),
            total_sources=len(seen),
        )
        report = state["report"].model_copy(update={"source_diversity": diversity})
        return {"report": report}
    except Exception:
        logger.exception("Graph: [diversity] node failed; skipping.")
        return {}


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
    graph.add_node("contradict", contradict_node)
    graph.add_node("diversity", diversity_node)

    graph.add_edge(START, "plan")

    for current, next_node in [
        ("plan",       "research"),
        ("research",   "write"),
        ("write",      "fact_check"),
        ("fact_check", "cite"),
        ("cite",       "contradict"),
        ("contradict", "diversity"),
    ]:
        graph.add_conditional_edges(
            current,
            _should_continue,
            {"continue": next_node, "end": END},
        )

    graph.add_edge("diversity", END)

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