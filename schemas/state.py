from typing import Optional
from typing_extensions import TypedDict

from schemas.research import ResearchDocument, ResearchPlan
from schemas.report import FinalReport


class GraphState(TypedDict, total=False):
    """
    Shared state object passed between every node in the LangGraph pipeline.

    Uses TypedDict (not Pydantic BaseModel) because LangGraph merges node
    return dicts directly into the state object.

    Fields
    ------
    query    : Original user query (required).
    audience : Optional audience override from the UI dropdown.
               If set, passed to PlannerAgent to constrain its output.
    depth    : Optional depth override from the UI dropdown.
               If set, passed to PlannerAgent to constrain its output.
    plan     : Populated by PlannerAgent.
    document : Populated by ResearcherAgent.
    draft    : Populated by WriterAgent (pre fact-check).
    report   : Final enriched report set by CitationAgent.
    error    : Set by any node that catches an exception.
    """

    query: str
    audience: Optional[str]
    depth: Optional[str]
    plan: Optional[ResearchPlan]
    document: Optional[ResearchDocument]
    draft: Optional[FinalReport]
    report: Optional[FinalReport]
    error: Optional[str]