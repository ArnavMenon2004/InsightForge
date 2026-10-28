from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from agents.base_agent import BaseAgent
from config.llm import GEMINI_MODEL
from config.settings import SECTION_RATE_LIMIT_DELAY
from prompts.writer import (
    REPORT_METADATA_PROMPT,
    SECTION_WRITER_PROMPT,
    STUDENT_GUIDE_PROMPT,
)
from schemas.report import FinalReport
from schemas.research import ResearchDocument, ResearchSection, StudentGuidance
from schemas.result import AgentResult


class _SectionDraft(BaseModel):
    """Structured output for one polished section body."""

    body: str = Field(
        description=(
            "Polished prose body for this section. "
            "No heading — heading is added by the system. "
            "Minimum 3 paragraphs, 150–250 words."
        )
    )


class _ReportMetadata(BaseModel):
    """Structured output for report title and executive summary."""

    title: str = Field(
        description="Precise, engaging report title. Plain text, no Markdown."
    )

    executive_summary: str = Field(
        description=(
            "Single self-contained paragraph, 60–100 words. "
            "Plain text only."
        )
    )


class WriterAgent(BaseAgent):
    """
    Transforms a ResearchDocument into a polished Markdown report.

    Strategy — section-by-section writing
    --------------------------------------
    Instead of sending all section summaries in one massive prompt
    (which exceeds token limits with rich content), the WriterAgent
    makes one focused LLM call per section, then assembles the Markdown
    body, then makes one final call to produce the title and executive
    summary.

    This means:
    - Each individual LLM call stays small and well under token limits.
    - Each section gets the LLM's full attention rather than sharing
      a context window with five other sections.
    - Output quality improves because the prompt is focused.

    Model: Gemini via _run_gemini_chain — reliable structured output,
    no tool_use_failed errors, 500 RPD on free tier.
    """

    def __init__(self) -> None:
        super().__init__("Writer Agent")

    def run(self, document: ResearchDocument) -> AgentResult:
        """
        Write a full polished report from the research document.

        Args:
            document: ResearchDocument from ResearcherAgent.

        Returns:
            AgentResult whose data is a FinalReport (fact_check and
            citations are None at this stage).
        """
        return self.execute(document=document)

    def execute(self, **kwargs) -> AgentResult:  # type: ignore[override]
        document: ResearchDocument = kwargs["document"]

        self.logger.info(
            f"{self.name}: Writing {len(document.sections)} sections."
        )

        start = time.perf_counter()

        # Step 1 — Write each section independently.
        section_bodies: list[tuple[str, str]] = []  # (title, polished_body)

        for index, section in enumerate(document.sections, start=1):
            self.logger.info(
                f"{self.name}: Writing section {index}/{len(document.sections)} "
                f"— '{section.title}'"
            )

            if index > 1 and SECTION_RATE_LIMIT_DELAY > 0:
                time.sleep(SECTION_RATE_LIMIT_DELAY)

            draft = self._write_section(section, document)
            section_bodies.append((section.title, draft.body))

        # Step 2 — Assemble the full Markdown body.
        markdown = self._assemble_markdown(section_bodies)

        # Step 3 — Generate title and executive summary from the full body.
        self.logger.info(f"{self.name}: Generating title and executive summary.")

        if SECTION_RATE_LIMIT_DELAY > 0:
            time.sleep(SECTION_RATE_LIMIT_DELAY)

        metadata = self._generate_metadata(document, markdown)

        # Step 4 — Student Guide Mode: per-section research guidance.
        student_guidance: dict[str, StudentGuidance] | None = None
        if document.audience == "Student":
            student_guidance = self._generate_student_guidance(document)

        report = FinalReport(
            title=metadata.title,
            executive_summary=metadata.executive_summary,
            markdown=markdown,
            student_guidance=student_guidance,
        )

        elapsed = time.perf_counter() - start
        self.logger.info(f"{self.name}: Report written in {elapsed:.2f}s.")

        return AgentResult(
            success=True,
            data=report,
            model=GEMINI_MODEL,
            execution_time=elapsed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_section(
        self,
        section: ResearchSection,
        document: ResearchDocument,
    ) -> _SectionDraft:
        """
        Ask Gemini to write polished prose for one section.

        Args:
            section:  The ResearchSection to write.
            document: Parent document (provides topic, audience, depth).

        Returns:
            A _SectionDraft containing the polished body text.
        """
        return self._run_gemini_chain(
            schema=_SectionDraft,
            prompt=SECTION_WRITER_PROMPT,
            inputs={
                "topic": document.topic,
                "audience": document.audience,
                "depth": document.depth,
                "section_title": section.title,
                "section_summary": section.summary,
            },
        )

    def _generate_metadata(
        self,
        document: ResearchDocument,
        report_body: str,
    ) -> _ReportMetadata:
        """
        Ask Gemini to produce the report title and executive summary.

        Args:
            document:    Parent document for context.
            report_body: The fully assembled Markdown body.

        Returns:
            A _ReportMetadata with title and executive_summary.
        """
        return self._run_gemini_chain(
            schema=_ReportMetadata,
            prompt=REPORT_METADATA_PROMPT,
            inputs={
                "topic": document.topic,
                "audience": document.audience,
                "depth": document.depth,
                "report_body": report_body,
            },
        )

    def _generate_student_guidance(
        self,
        document: ResearchDocument,
    ) -> dict[str, StudentGuidance]:
        """
        Produce a StudentGuidance block for every section.

        Returns:
            Map of section title -> StudentGuidance.
        """
        self.logger.info(f"{self.name}: Generating student guidance.")

        def _one(position: int, section: ResearchSection) -> StudentGuidance | None:
            # Staggered start keeps SECTION_RATE_LIMIT_DELAY spacing between
            # call starts while the calls themselves overlap.
            if SECTION_RATE_LIMIT_DELAY > 0:
                time.sleep(SECTION_RATE_LIMIT_DELAY * (position + 1))
            try:
                return self._run_gemini_chain(
                    schema=StudentGuidance,
                    prompt=STUDENT_GUIDE_PROMPT,
                    inputs={
                        "section_title": section.title,
                        "section_summary": section.summary,
                    },
                )
            except Exception:
                # Guidance is optional; don't lose the whole report over one section.
                self.logger.exception(
                    f"{self.name}: Student guidance failed for '{section.title}'; skipping."
                )
                return None

        with ThreadPoolExecutor(max_workers=len(document.sections) or 1) as pool:
            results = list(pool.map(_one, range(len(document.sections)), document.sections))

        return {
            section.title: g
            for section, g in zip(document.sections, results)
            if g is not None
        }

    @staticmethod
    def _assemble_markdown(section_bodies: list[tuple[str, str]]) -> str:
        """
        Combine section titles and polished bodies into a Markdown document.

        Paragraph breaks are normalised: sentences are grouped into
        paragraphs of 3 sentences each for visual breathing room.

        Args:
            section_bodies: List of (title, body) tuples in order.

        Returns:
            Complete Markdown string starting with the first ## heading.
        """
        import re

        blocks: list[str] = []

        for title, body in section_bodies:
            normalised = re.sub(r"\r?\n", " ", body).strip()

            sentences = re.split(r"(?<=\.)\s+", normalised)
            sentences = [s.strip() for s in sentences if s.strip()]

            paragraphs: list[str] = []
            chunk_size = 3
            for i in range(0, len(sentences), chunk_size):
                chunk = sentences[i:i + chunk_size]
                paragraphs.append(" ".join(chunk))

            body_text = "\n\n".join(paragraphs) if paragraphs else normalised
            blocks.append(f"## {title}\n\n{body_text}")

        return "\n\n---\n\n".join(blocks)