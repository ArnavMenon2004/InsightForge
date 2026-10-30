from __future__ import annotations

import re

from agents.base_agent import BaseAgent
from config.settings import FC_MAX_SOURCES, FC_SNIPPET_CHARS
from prompts.fact_checker import FACT_CHECKER_PROMPT
from schemas.report import FinalReport, FactCheckReport, SectionConfidence
from schemas.research import ResearchDocument
from schemas.result import AgentResult
from schemas.source import Source


class FactCheckerAgent(BaseAgent):
    """
    Verifies factual claims in the writer's draft against the original sources.

    Sources are capped at FC_MAX_SOURCES and snippets trimmed to
    FC_SNIPPET_CHARS before being sent to the LLM.  This keeps the
    single prompt small even when the researcher has collected 30–40
    enriched sources.
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

        fact_check = self._with_overall_confidence(result.data)

        enriched = draft.model_copy(
            update={
                "fact_check": fact_check,
                "section_confidences": self._compute_section_confidences(
                    draft.markdown, fact_check
                ),
            }
        )

        return AgentResult(
            success=True,
            data=enriched,
            model=result.model,
            execution_time=result.execution_time,
        )

    @staticmethod
    def _with_overall_confidence(fact_check: FactCheckReport) -> FactCheckReport:
        """
        Set overall_confidence from the verdicts (High >=80% SUPPORTED,
        Medium >=50%, else Low), the rule the prompt states.  The model
        sometimes miscounts, so the label is derived here instead.
        """
        if not fact_check.results:
            return fact_check
        share = sum(r.verdict == "SUPPORTED" for r in fact_check.results) / len(fact_check.results)
        label = "High" if share >= 0.8 else "Medium" if share >= 0.5 else "Low"
        return fact_check.model_copy(update={"overall_confidence": label})

    @staticmethod
    def _compute_section_confidences(
        markdown: str,
        fact_check: FactCheckReport,
    ) -> dict[str, SectionConfidence]:
        """
        Compute the share of SUPPORTED claims per report section.

        Sections are the ``\\n---\\n``-separated blocks of ``markdown``.  A
        FactCheckResult belongs to a section when its claim text appears
        in that block (whitespace- and case-insensitive).
        """

        def norm(text: str) -> str:
            return " ".join(text.split()).lower()

        def tokens(text: str) -> set[str]:
            return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3}

        blocks: list[tuple[str, str]] = []  # (title, normalised block text)
        for idx, raw in enumerate(re.split(r"\n---\n", markdown), start=1):
            raw = raw.strip()
            if not raw:
                continue
            m = re.match(r"^##\s+(.+)", raw)
            blocks.append((m.group(1).strip() if m else f"Section {idx}", norm(raw)))

        # Assign each claim to one section: verbatim match first, otherwise
        # the section covering most of the claim's words (LLMs paraphrase).
        assigned: dict[str, list] = {title: [] for title, _ in blocks}
        for r in fact_check.results:
            if not r.claim.strip():
                continue
            hit = next((t for t, b in blocks if norm(r.claim) in b), None)
            if hit is None:
                words = tokens(r.claim)
                if words:
                    best_title, best = max(
                        ((t, len(words & tokens(b)) / len(words)) for t, b in blocks),
                        key=lambda x: x[1],
                    )
                    if best >= 0.6:
                        hit = best_title
            if hit is not None:
                assigned[hit].append(r)

        confidences: dict[str, SectionConfidence] = {}

        for title, _ in blocks:
            claims = assigned[title]
            total = len(claims)
            supported = sum(1 for r in claims if r.verdict == "SUPPORTED")
            pct = supported / total * 100 if total else 100.0
            label = "High" if pct >= 70 else "Medium" if pct >= 40 else "Low"

            confidences[title] = SectionConfidence(
                section_title=title,
                supported=supported,
                total=total,
                confidence_pct=round(pct, 1),
                confidence_label=label,
            )

        return confidences

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
        to keep the total prompt small.
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