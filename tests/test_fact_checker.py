"""Unit tests for FactCheckerAgent."""

from unittest.mock import patch

from agents.fact_checker import FactCheckerAgent
from schemas.report import FinalReport, FactCheckReport, FactCheckResult
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


def _mock_fact_check() -> FactCheckReport:
    return FactCheckReport(
        results=[
            FactCheckResult(
                claim="Qubits enable superposition.",
                verdict="SUPPORTED",
                explanation="Directly stated in source [1].",
            )
        ],
        overall_confidence="High",
    )


class TestFactCheckerAgent:

    def test_run_attaches_fact_check_to_draft(self):
        agent = FactCheckerAgent()
        with patch.object(agent, "execute") as mock_exec:
            mock_exec.return_value = AgentResult(
                success=True,
                data=_mock_fact_check(),
                model="test",
                execution_time=0.1,
            )
            result = agent.run(_make_draft(), _make_document())

        assert result.success is True
        report: FinalReport = result.data
        assert report.fact_check is not None
        assert report.fact_check.overall_confidence == "High"
        assert len(report.fact_check.results) == 1

    def test_collect_all_sources_deduplicates(self):
        doc = _make_document()
        # Add same URL to a second section
        doc.sections.append(
            ResearchSection(
                title="Section 2",
                summary="More content.",
                sources=[doc.sections[0].sources[0]],
            )
        )
        sources = FactCheckerAgent._collect_all_sources(doc)
        assert len(sources) == 1

    def test_original_draft_fields_preserved(self):
        agent = FactCheckerAgent()
        with patch.object(agent, "execute") as mock_exec:
            mock_exec.return_value = AgentResult(
                success=True,
                data=_mock_fact_check(),
                model="test",
                execution_time=0.1,
            )
            result = agent.run(_make_draft(), _make_document())

        report: FinalReport = result.data
        assert report.title == "Quantum Report"
        assert report.markdown == "# Quantum\n\nQubits enable superposition."
