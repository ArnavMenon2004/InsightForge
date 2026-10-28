from __future__ import annotations

import time

from agents.base_agent import BaseAgent
from config.llm import GEMINI_MODEL
from prompts.contradiction import CONTRADICTION_PROMPT
from schemas.report import ContradictionReport, FinalReport
from schemas.research import ResearchDocument
from schemas.result import AgentResult


class ContradictionAgent(BaseAgent):
    """
    Scans all sections for claims that contradict each other across sections.

    The result is attached to ``FinalReport.contradiction_report``.
    """

    def __init__(self) -> None:
        super().__init__("Contradiction Agent")

    def run(self, report: FinalReport, document: ResearchDocument) -> AgentResult:
        return self.execute(report=report, document=document)

    def execute(self, **kwargs) -> AgentResult:  # type: ignore[override]
        report: FinalReport = kwargs["report"]
        document: ResearchDocument = kwargs["document"]

        start = time.perf_counter()

        sections_text = "\n\n".join(
            f"## {s.title}\n{s.summary[:500]}" for s in document.sections
        )

        result: ContradictionReport = self._run_gemini_chain(
            schema=ContradictionReport,
            prompt=CONTRADICTION_PROMPT,
            inputs={"sections_text": sections_text},
        )

        n = len(result.contradictions)
        result.consistency_score = "High" if n == 0 else "Medium" if n <= 2 else "Low"

        enriched = report.model_copy(update={"contradiction_report": result})

        return AgentResult(
            success=True,
            data=enriched,
            model=GEMINI_MODEL,
            execution_time=time.perf_counter() - start,
        )
