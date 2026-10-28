from pydantic import BaseModel, Field

from schemas.source import Source


class ResearchPlan(BaseModel):
    """
    Output produced by the PlannerAgent.

    Contains everything downstream agents need to know about the
    research job: topic, audience, depth, seed keywords, and the
    ordered list of sections to research.
    """

    topic: str = Field(
        description="Main research topic."
    )

    audience: str = Field(
        description="Target audience: Beginner, Student, Developer, or Executive."
    )

    depth: str = Field(
        description="Research depth: Basic, Intermediate, or Advanced."
    )

    keywords: list[str] = Field(
        description="Seed keywords for web search."
    )

    sections: list[str] = Field(
        description="Ordered list of section titles to research."
    )


class SearchQueries(BaseModel):
    """
    Focused web search queries generated for a single report section.

    The ResearcherAgent asks the LLM to produce this for every section
    in the ResearchPlan before calling WebSearchTool.
    """

    queries: list[str] = Field(
        description=(
            "Two to four focused web search queries for this section. "
            "Each query should target a distinct angle of the topic."
        )
    )


class ResearchSection(BaseModel):
    """
    Researched content for one section of the report.

    Produced by the ResearcherAgent after searching the web and
    summarising the retrieved sources with the LLM.
    """

    title: str = Field(
        description="Section title, matching the title from the ResearchPlan."
    )

    summary: str = Field(
        description=(
            "Comprehensive prose summary of everything found for this "
            "section.  Written in a neutral, informative tone suitable "
            "for the target audience."
        )
    )

    sources: list[Source] = Field(
        description="Normalised Source objects retrieved during research."
    )


class ResearchDocument(BaseModel):
    """
    Complete research output produced by the ResearcherAgent.

    Passed downstream to the WriterAgent, FactCheckerAgent, and
    CitationAgent.
    """

    topic: str
    audience: str
    depth: str
    sections: list[ResearchSection]


class StudentGuidance(BaseModel):
    """Guidance block for student audience, one per section."""

    search_terms: list[str] = Field(
        description="2-4 specific search terms the student can use to find more on this section."
    )

    source_types: list[str] = Field(
        description="Types of sources most useful (e.g. arXiv, IEEE Xplore, textbooks)."
    )

    related_subtopics: list[str] = Field(
        description="2-3 related subtopics worth exploring after reading this section."
    )


class SectionSummary(BaseModel):
    """
    Structured LLM output for a single section's prose summary.

    Used as the ``output_schema`` when the ResearcherAgent calls the
    summarisation chain.  The ``summary`` field is then transferred
    into the final ``ResearchSection`` object alongside its sources.
    """

    summary: str = Field(
        description=(
            "Comprehensive prose summary synthesised from the provided "
            "source material.  Neutral, informative, and audience-appropriate."
        )
    )
