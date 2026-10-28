from __future__ import annotations

from pydantic import BaseModel

from agents.base_agent import BaseAgent
from config.settings import FC_MAX_SOURCES
from prompts.citation import CITATION_PROMPT
from schemas.report import CitedClaim, FinalReport
from schemas.research import ResearchDocument
from schemas.result import AgentResult
from schemas.source import Source


class _CitationResult(BaseModel):
    """Structured output: just the citation list."""
    citations: list[CitedClaim]


class CitationAgent(BaseAgent):
    """
    Maps factual claims in the report to their supporting sources.

    Source list is capped at FC_MAX_SOURCES (titles + URLs only, no
    snippets) to keep the prompt small.
    """

    prompt = CITATION_PROMPT
    output_schema = _CitationResult

    def __init__(self) -> None:
        super().__init__("Citation Agent")

    def run(
        self,
        draft: FinalReport,
        document: ResearchDocument,
    ) -> AgentResult:
        all_sources = self._collect_all_sources(document)
        sources_index = self._format_sources_index(
            all_sources,
            max_sources=FC_MAX_SOURCES,
        )

        result = self.execute(
            report_markdown=draft.markdown,
            sources_index=sources_index,
        )

        final_report = draft.model_copy(
            update={"citations": result.data.citations}
        )

        return AgentResult(
            success=True,
            data=final_report,
            model=result.model,
            execution_time=result.execution_time,
        )

    @staticmethod
    def _collect_all_sources(document: ResearchDocument) -> list[Source]:
        """Flatten and deduplicate all sources across every section."""
        seen: set[str] = set()
        sources: list[Source] = []
        for section in document.sections:
            for source in section.sources:
                url_key = str(source.url)
                if url_key not in seen:
                    seen.add(url_key)
                    sources.append(source)
        return sources

    @staticmethod
    def _format_sources_index(
        sources: list[Source],
        max_sources: int,
    ) -> str:
        """
        Render sources as a numbered index (title + URL only, no snippets).

        The citation agent only needs to match claims to source titles —
        it does not need full content, so snippets are omitted entirely
        to keep the prompt small.
        """
        lines: list[str] = []
        for index, source in enumerate(sources[:max_sources], start=1):
            lines.append(f"[{index}] {source.title} — {source.url}")
        return "\n".join(lines)