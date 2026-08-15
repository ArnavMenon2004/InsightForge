from __future__ import annotations

from agents.base_agent import BaseAgent
from config.settings import FC_MAX_SOURCES, FC_SNIPPET_CHARS
from prompts.fact_checker import FACT_CHECKER_PROMPT
from schemas.report import FinalReport, FactCheckReport
from schemas.research import ResearchDocument
from schemas.result import AgentResult
from schemas.source import Source


class FactCheckerAgent(BaseAgent):
    """
    Verifies factual claims in the writer's draft against the original sources.

    Sources are capped at FC_MAX_SOURCES and snippets trimmed to
    FC_SNIPPET_CHARS before being sent to the LLM.  This keeps the
    single prompt well within Groq's 12,000 TPM per-request limit
    even when the researcher has collected 30–40 enriched sources.
    """

    prompt = FACT_CHECKER_PROMPT
    output_schema = FactCheckReport

    def __init__(self) -> None:
        super().__init__("Fact Checker Agent")

    def run(
        self,
        draft: FinalReport,
        document: ResearchDocument,
    ) -> AgentResult:
        all_sources = self._collect_all_sources(document)
        sources_text = self._format_sources(
            all_sources,
            max_sources=FC_MAX_SOURCES,
            snippet_chars=FC_SNIPPET_CHARS,
        )

        result = self.execute(
            report_markdown=draft.markdown,
            sources_text=sources_text,
        )

        enriched = draft.model_copy(
            update={"fact_check": result.data}
        )

        return AgentResult(
            success=True,
            data=enriched,
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
    def _format_sources(
        sources: list[Source],
        max_sources: int,
        snippet_chars: int,
    ) -> str:
        """
        Render sources as a numbered block for the fact-checker prompt.

        Caps at ``max_sources`` and trims each snippet to ``snippet_chars``
        to keep the total prompt within Groq's per-request token limit.
        """
        blocks: list[str] = []
        for index, source in enumerate(sources[:max_sources], start=1):
            block = (
                f"[{index}] {source.title}\n"
                f"URL    : {source.url}\n"
                f"Content: {source.snippet[:snippet_chars]}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)