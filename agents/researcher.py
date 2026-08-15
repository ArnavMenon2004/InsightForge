from __future__ import annotations

import time

from agents.base_agent import BaseAgent
from config.llm import SUMMARISER_MODEL
from config.settings import (
    MAX_RESULTS_PER_QUERY,
    MAX_SOURCES_FOR_PROMPT,
    SECTION_RATE_LIMIT_DELAY,
    SNIPPET_CHAR_LIMIT,
)
from prompts.researcher import QUERY_GENERATION_PROMPT, SECTION_SUMMARY_PROMPT
from schemas.research import (
    ResearchDocument,
    ResearchPlan,
    ResearchSection,
    SearchQueries,
    SectionSummary,
)
from schemas.result import AgentResult
from schemas.source import Source
from tools.web_fetch import WebFetchTool
from tools.web_search import WebSearchTool


class ResearcherAgent(BaseAgent):
    """
    Conducts web research for every section in a ResearchPlan.

    Workflow (per section)
    ----------------------
    1. Ask gemma2-9b-it to generate focused search queries.
    2. Execute each query via WebSearchTool → Source objects.
    3. Enrich each source: attempt to fetch full page content via
       WebFetchTool; fall back to Tavily snippet if fetch fails.
    4. Deduplicate sources by URL.
    5. Ask gemma2-9b-it to summarise the enriched content into prose.
    6. Assemble a ResearchSection from the summary and sources.

    Uses gemma2-9b-it (via _run_summariser_chain) for all high-volume
    per-section calls.  This model has a higher TPM limit than llama-3.3-70b
    and uses fewer tokens per output word, giving richer summaries within
    the free-tier budget.

    Args:
        search_tool: Optional WebSearchTool. Defaults to a new instance.
        fetch_tool:  Optional WebFetchTool.  Defaults to a new instance.
    """

    def __init__(
        self,
        search_tool: WebSearchTool | None = None,
        fetch_tool: WebFetchTool | None = None,
    ) -> None:
        super().__init__("Researcher Agent")
        self._search_tool = search_tool or WebSearchTool()
        self._fetch_tool = fetch_tool or WebFetchTool()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, plan: ResearchPlan) -> AgentResult:
        """Research every section in the plan and return a ResearchDocument."""
        return self.execute(plan=plan)

    def execute(self, **kwargs) -> AgentResult:  # type: ignore[override]
        plan: ResearchPlan = kwargs["plan"]

        self.logger.info(
            f"{self.name}: Starting research for '{plan.topic}' "
            f"({len(plan.sections)} sections)."
        )

        start = time.perf_counter()
        sections: list[ResearchSection] = []

        for index, section_title in enumerate(plan.sections, start=1):
            self.logger.info(
                f"{self.name}: Section {index}/{len(plan.sections)} — "
                f"'{section_title}'"
            )

            if index > 1 and SECTION_RATE_LIMIT_DELAY > 0:
                self.logger.debug(
                    f"{self.name}: Rate-limit pause {SECTION_RATE_LIMIT_DELAY}s."
                )
                time.sleep(SECTION_RATE_LIMIT_DELAY)

            section = self._research_section(section_title, plan)
            sections.append(section)

        document = ResearchDocument(
            topic=plan.topic,
            audience=plan.audience,
            depth=plan.depth,
            sections=sections,
        )

        elapsed = time.perf_counter() - start
        self.logger.info(
            f"{self.name}: Research complete in {elapsed:.2f}s — "
            f"{len(sections)} sections assembled."
        )

        return AgentResult(
            success=True,
            data=document,
            model=SUMMARISER_MODEL,
            execution_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _research_section(
        self,
        section_title: str,
        plan: ResearchPlan,
    ) -> ResearchSection:
        """Run the full research pipeline for one section."""
        queries = self._generate_queries(section_title, plan)
        self.logger.info(
            f"{self.name}: {len(queries.queries)} queries for '{section_title}'."
        )

        sources = self._collect_and_enrich_sources(queries)
        self.logger.info(
            f"{self.name}: {len(sources)} unique sources for '{section_title}'."
        )

        summary = self._summarise_section(section_title, sources, plan)

        return ResearchSection(
            title=section_title,
            summary=summary.summary,
            sources=sources,
        )

    def _generate_queries(
        self,
        section_title: str,
        plan: ResearchPlan,
    ) -> SearchQueries:
        """
        Ask gemma2-9b-it to generate focused search queries for one section.
        Uses _run_summariser_chain for higher TPM headroom.
        """
        return self._run_summariser_chain(
            schema=SearchQueries,
            prompt=QUERY_GENERATION_PROMPT,
            inputs={
                "section_title": section_title,
                "topic": plan.topic,
                "audience": plan.audience,
                "depth": plan.depth,
                "keywords": ", ".join(plan.keywords),
            },
        )

    def _collect_and_enrich_sources(
        self,
        queries: SearchQueries,
        max_results_per_query: int = MAX_RESULTS_PER_QUERY,
    ) -> list[Source]:
        """
        Execute every query, deduplicate results, then enrich each source
        by fetching its full page content.

        Enrichment replaces the short Tavily snippet with up to
        SNIPPET_CHAR_LIMIT characters of real article body text.
        If the fetch fails or returns nothing, the original Tavily
        snippet is kept as a fallback — the pipeline never hard-fails
        due to a fetch error.

        Args:
            queries:               SearchQueries to execute.
            max_results_per_query: Results requested per query.

        Returns:
            Deduplicated, enriched list of Source objects.
        """
        seen_urls: set[str] = set()
        sources: list[Source] = []

        for query in queries.queries:
            results = self._search_tool.execute(
                query=query,
                max_results=max_results_per_query,
            )
            for source in results:
                url_key = str(source.url)
                if url_key not in seen_urls:
                    seen_urls.add(url_key)
                    sources.append(source)

        # Enrich: replace snippet with fetched full-page content.
        enriched: list[Source] = []
        for source in sources:
            fetched = self._fetch_tool.fetch(
                url=str(source.url),
                max_chars=SNIPPET_CHAR_LIMIT,
            )
            if fetched:
                enriched.append(
                    source.model_copy(update={"snippet": fetched})
                )
                self.logger.debug(
                    f"{self.name}: Enriched '{source.title}' "
                    f"({len(fetched)} chars)."
                )
            else:
                # Fallback: trim original Tavily snippet.
                enriched.append(
                    source.model_copy(
                        update={"snippet": source.snippet[:SNIPPET_CHAR_LIMIT]}
                    )
                )

        return enriched

    def _summarise_section(
        self,
        section_title: str,
        sources: list[Source],
        plan: ResearchPlan,
    ) -> SectionSummary:
        """
        Ask gemma2-9b-it to synthesise sources into a prose summary.

        Uses _run_summariser_chain for higher TPM headroom.
        Falls back to a placeholder if no sources are available.
        """
        if not sources:
            self.logger.warning(
                f"{self.name}: No sources for '{section_title}'. "
                "Inserting placeholder."
            )
            return SectionSummary(
                summary=(
                    f"[No sources were retrieved for '{section_title}'. "
                    "This section requires manual review.]"
                )
            )

        sources_text = self._format_sources_for_prompt(
            sources[:MAX_SOURCES_FOR_PROMPT]
        )

        return self._run_summariser_chain(
            schema=SectionSummary,
            prompt=SECTION_SUMMARY_PROMPT,
            inputs={
                "section_title": section_title,
                "topic": plan.topic,
                "audience": plan.audience,
                "depth": plan.depth,
                "sources_text": sources_text,
            },
        )

    @staticmethod
    def _format_sources_for_prompt(sources: list[Source]) -> str:
        """
        Render Source objects into a numbered plain-text block for the prompt.
        """
        blocks: list[str] = []
        for index, source in enumerate(sources, start=1):
            block = (
                f"[{index}] {source.title}\n"
                f"URL    : {source.url}\n"
                f"Content: {source.snippet}"
            )
            blocks.append(block)
        return "\n\n".join(blocks)