from __future__ import annotations

import time
from abc import ABC
from typing import Any

from config.llm import (
    GEMINI_MODEL,
    GROQ_MODEL,
    get_structured_gemini_llm,
    get_structured_llm,
    get_structured_summariser_llm,
)
from schemas.result import AgentResult
from utils.logger import logger


class BaseAgent(ABC):
    """
    Abstract base class shared by every agent.

    Provider routing
    ----------------
    - ``execute()``              → Gemini (default for most agents)
    - ``_run_chain()``           → Groq llama-3.3-70b (WriterAgent only)
    - ``_run_gemini_chain()``    → Gemini gemini-2.0-flash
    - ``_run_summariser_chain()``→ Gemini (alias for ResearcherAgent)

    WriterAgent overrides ``execute()`` and calls ``_run_chain()``
    explicitly to use Groq for prose quality.  All other agents use
    the default ``execute()`` which routes to Gemini.
    """

    prompt = None
    output_schema = None

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logger

    def execute(self, **kwargs: Any) -> AgentResult:
        """
        Run the agent's prompt through GEMINI and return an AgentResult.

        Default for PlannerAgent, FactCheckerAgent, CitationAgent.
        WriterAgent overrides this to use Groq instead.
        """
        if self.prompt is None:
            raise ValueError(f"{self.name}: prompt is not set.")
        if self.output_schema is None:
            raise ValueError(f"{self.name}: output_schema is not set.")

        self.logger.info(f"{self.name}: Starting execution (Gemini).")
        start = time.perf_counter()

        try:
            llm = get_structured_gemini_llm(self.output_schema)
            chain = self.prompt | llm
            response = chain.invoke(kwargs)
            elapsed = time.perf_counter() - start

            self.logger.info(f"{self.name}: Finished in {elapsed:.2f}s.")

            return AgentResult(
                success=True,
                data=response,
                model=GEMINI_MODEL,
                execution_time=elapsed,
            )

        except Exception:
            self.logger.exception(f"{self.name}: Execution failed.")
            raise

    def _run_chain(
        self,
        schema: type,
        prompt: Any,
        inputs: dict[str, Any],
    ) -> Any:
        """
        Execute a prompt through GROQ (llama-3.3-70b).
        Used exclusively by WriterAgent for highest prose quality.
        """
        self.logger.debug(f"{self.name}: _run_chain [Groq] → {schema.__name__}")
        llm = get_structured_llm(schema)
        chain = prompt | llm
        return chain.invoke(inputs)

    def _run_gemini_chain(
        self,
        schema: type,
        prompt: Any,
        inputs: dict[str, Any],
    ) -> Any:
        """
        Execute a prompt through GEMINI (gemini-2.0-flash).
        Used by ResearcherAgent and any agent needing explicit Gemini routing.
        """
        self.logger.debug(f"{self.name}: _run_gemini_chain [Gemini] → {schema.__name__}")
        llm = get_structured_gemini_llm(schema)
        chain = prompt | llm
        return chain.invoke(inputs)

    def _run_summariser_chain(
        self,
        schema: type,
        prompt: Any,
        inputs: dict[str, Any],
    ) -> Any:
        """
        Execute a prompt through the summariser LLM (now Gemini).
        Alias kept for ResearcherAgent compatibility.
        """
        self.logger.debug(f"{self.name}: _run_summariser_chain [Gemini] → {schema.__name__}")
        llm = get_structured_summariser_llm(schema)
        chain = prompt | llm
        return chain.invoke(inputs)