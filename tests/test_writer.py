"""Unit tests for WriterAgent."""

from unittest.mock import patch

from agents.writer import WriterAgent
from schemas.report import FinalReport
from schemas.research import ResearchDocument, ResearchSection
from schemas.source import Source, SourceType


def _make_document() -> ResearchDocument:
    return ResearchDocument(
        topic="Quantum Computing",
        audience="Developer",
        depth="Intermediate",
        sections=[
            ResearchSection(
                title="Introduction",
                summary="Quantum computing uses qubits.",
                sources=[
                    Source(
                        title="Qubit Basics",
                        url="https://example.com/qubits",
                        snippet="Qubits are the basic unit.",
                        source_type=SourceType.OTHER,
                    )
                ],
            )
        ],
    )


def _mock_report() -> FinalReport:
    return FinalReport(
        title="Quantum Computing Report",
        executive_summary="A concise overview.",
        markdown="# Quantum Computing\n\n## Introduction\n\nContent here.",
    )


class TestWriterAgent:

    def test_run_returns_final_report(self):
        agent = WriterAgent()
        with patch.object(agent, "execute") as mock_exec:
            from schemas.result import AgentResult
            mock_exec.return_value = AgentResult(
                success=True,
                data=_mock_report(),
                model="test",
                execution_time=0.1,
            )
            result = agent.run(_make_document())

        assert result.success is True
        assert isinstance(result.data, FinalReport)
        assert result.data.title == "Quantum Computing Report"

    def test_format_sections_joins_with_divider(self):
        sections = [
            ResearchSection(
                title="A",
                summary="Summary A.",
                sources=[],
            ),
            ResearchSection(
                title="B",
                summary="Summary B.",
                sources=[],
            ),
        ]
        text = WriterAgent._format_sections(sections)
        assert "## A" in text
        assert "## B" in text
        assert "---" in text
        assert "Summary A." in text

    def test_fact_check_and_citations_start_as_none(self):
        """FinalReport from WriterAgent should have null fact_check and citations."""
        report = _mock_report()
        assert report.fact_check is None
        assert report.citations is None
