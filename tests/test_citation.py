"""Unit tests for CitationAgent."""

from unittest.mock import patch

from agents.citation import CitationAgent
from schemas.report import CitedClaim, FinalReport, _CitationResult
from schemas.research import ResearchDocument, ResearchSection
from schemas.result import AgentResult
from schemas.source import Source, SourceType


def _make_draft() -> FinalReport:
    return FinalReport(
        title="Quantum Report",
        executive_summary="Overview.",
        markdown="# Quantum\n\nQubits enable superposition.",
    )


def _make_document() -> ResearchDocument:
    return ResearchDocument(
        topic="Quantum Computing",
        audience="Developer",
        depth="Intermediate",
        sections=[
            ResearchSection(
                title="Intro",
                summary="Qubits.",
                sources=[
                    Source(
                        title="Qubit Source",
                        url="https://example.com/q",
                        snippet="Qubits are two-state systems.",
                        source_type=SourceType.OTHER,
                    )
                ],
            )
        ],
    )


class TestCitationAgent:

    def test_run_attaches_citations_to_report(self):
        agent = CitationAgent()
        with patch.object(agent, "execute") as mock_exec:
            mock_exec.return_value = AgentResult(
                success=True,
                data=_CitationResult(
                    citations=[
                        CitedClaim(
                            claim="Qubits enable superposition.",
                            source_indices=[1],
                        )
                    ]
                ),
                model="test",
                execution_time=0.1,
            )
            result = agent.run(_make_draft(), _make_document())

        assert result.success is True
        report: FinalReport = result.data
        assert report.citations is not None
        assert len(report.citations) == 1
        assert report.citations[0].source_indices == [1]

    def test_original_report_fields_preserved(self):
        agent = CitationAgent()
        with patch.object(agent, "execute") as mock_exec:
            mock_exec.return_value = AgentResult(
                success=True,
                data=_CitationResult(citations=[]),
                model="test",
                execution_time=0.1,
            )
            result = agent.run(_make_draft(), _make_document())

        report: FinalReport = result.data
        assert report.title == "Quantum Report"
        assert report.executive_summary == "Overview."

    def test_format_sources_index_numbering(self):
        sources = [
            Source(
                title="A",
                url="https://a.com",
                snippet="aa",
                source_type=SourceType.OTHER,
            ),
            Source(
                title="B",
                url="https://b.com",
                snippet="bb",
                source_type=SourceType.OTHER,
            ),
        ]
        index = CitationAgent._format_sources_index(sources)
        assert "[1]" in index
        assert "[2]" in index
        assert "https://a.com" in index
