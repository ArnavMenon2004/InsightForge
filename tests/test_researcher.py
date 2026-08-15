"""
Unit tests for ResearcherAgent.

All external calls (LLM + Tavily) are mocked so these tests run
without API keys and execute in milliseconds.
"""

from unittest.mock import MagicMock, patch

from agents.researcher import ResearcherAgent
from schemas.research import (
    ResearchDocument,
    ResearchPlan,
    ResearchSection,
    SearchQueries,
    SectionSummary,
)
from schemas.source import Source, SourceType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_plan() -> ResearchPlan:
    return ResearchPlan(
        topic="Quantum Computing",
        audience="Developer",
        depth="Intermediate",
        keywords=["qubits", "superposition", "entanglement"],
        sections=["Introduction", "Applications"],
    )


def _make_source(n: int = 1) -> Source:
    return Source(
        title=f"Source {n}",
        url=f"https://example.com/source-{n}",
        snippet=f"Snippet for source {n}.",
        source_type=SourceType.OTHER,
        score=0.9,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestResearcherAgent:

    def test_run_returns_agent_result_with_research_document(self):
        """
        ResearcherAgent.run() should return an AgentResult whose data is a
        ResearchDocument with one ResearchSection per plan section.
        """
        mock_tool = MagicMock()
        mock_tool.execute.return_value = [_make_source(1), _make_source(2)]

        agent = ResearcherAgent(search_tool=mock_tool)

        with (
            patch.object(
                agent,
                "_run_chain",
                side_effect=_mock_run_chain,
            )
        ):
            plan = _make_plan()
            result = agent.run(plan)

        assert result.success is True
        assert isinstance(result.data, ResearchDocument)
        assert len(result.data.sections) == len(plan.sections)

    def test_sections_have_correct_titles(self):
        """Each ResearchSection title must match the plan's section title."""
        mock_tool = MagicMock()
        mock_tool.execute.return_value = [_make_source()]

        agent = ResearcherAgent(search_tool=mock_tool)

        with patch.object(agent, "_run_chain", side_effect=_mock_run_chain):
            plan = _make_plan()
            result = agent.run(plan)

        titles = [s.title for s in result.data.sections]
        assert titles == plan.sections

    def test_sections_contain_sources(self):
        """Sources returned by the tool must appear in the ResearchSection."""
        mock_tool = MagicMock()
        mock_tool.execute.return_value = [_make_source(1), _make_source(2)]

        agent = ResearcherAgent(search_tool=mock_tool)

        with patch.object(agent, "_run_chain", side_effect=_mock_run_chain):
            result = agent.run(_make_plan())

        for section in result.data.sections:
            assert len(section.sources) > 0

    def test_empty_search_results_produce_placeholder(self):
        """
        If WebSearchTool returns nothing, the section summary should be a
        clearly marked placeholder rather than raising an exception.
        """
        mock_tool = MagicMock()
        mock_tool.execute.return_value = []

        agent = ResearcherAgent(search_tool=mock_tool)

        # Only the query generation chain will be called (no sources → no
        # summarisation chain call).
        with patch.object(
            agent,
            "_run_chain",
            return_value=SearchQueries(queries=["quantum basics"]),
        ):
            result = agent.run(_make_plan())

        for section in result.data.sections:
            assert "[No sources" in section.summary

    def test_url_deduplication(self):
        """
        Duplicate URLs across queries must be deduplicated before the
        summarisation step.
        """
        duplicate_source = _make_source(1)  # same URL every time

        mock_tool = MagicMock()
        mock_tool.execute.return_value = [duplicate_source]

        agent = ResearcherAgent(search_tool=mock_tool)

        seen_source_counts: list[int] = []

        def capturing_run_chain(schema, prompt, inputs):
            result = _mock_run_chain(schema, prompt, inputs)
            return result

        with patch.object(agent, "_run_chain", side_effect=capturing_run_chain):
            result = agent.run(_make_plan())

        for section in result.data.sections:
            # Despite multiple queries returning the same URL, only one source
            # should appear per section.
            assert len(section.sources) == 1

    def test_format_sources_for_prompt(self):
        """_format_sources_for_prompt should number sources correctly."""
        sources = [_make_source(1), _make_source(2)]
        text = ResearcherAgent._format_sources_for_prompt(sources)

        assert "[1]" in text
        assert "[2]" in text
        assert "Source 1" in text
        assert "https://example.com/source-1" in text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_run_chain(schema, prompt, inputs):
    """
    Side-effect function for ``_run_chain`` that returns a sensible dummy
    object for each schema type used by ResearcherAgent.
    """
    if schema is SearchQueries:
        return SearchQueries(queries=["mock query one", "mock query two"])

    if schema is SectionSummary:
        section_title = inputs.get("section_title", "Unknown")
        return SectionSummary(
            summary=f"Mock summary for section: {section_title}."
        )

    raise ValueError(f"Unexpected schema in mock: {schema}")
