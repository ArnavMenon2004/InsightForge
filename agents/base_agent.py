from __future__ import annotations

import time
from abc import ABC
from typing import Any

from config.llm import (
    GEMINI_MODEL,
    get_structured_gemini_llm,
    get_structured_summariser_llm,
)
from schemas.result import AgentResult
from utils.logger import logger


class BaseAgent(ABC):
    """
    Abstract base class shared by every agent.

    Every LLM call goes to Gemini (model set in ``config/llm.py``):
    - ``execute()``              → default structured call for an agent
    - ``_run_gemini_chain()``    → explicit structured call with a given prompt
    - ``_run_summariser_chain()``→ alias used by ResearcherAgent
    """

    prompt = None
    output_schema = None

    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logger

    _TRANSIENT_MARKERS = (
        "503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded",
        "500 INTERNAL", "internal error", "502", "504", "bad gateway",
        "timed out", "timeout", "deadline",
    )
    _RETRY_WAITS = (15, 30, 60, 90)  # seconds before retries 2..5

    def _invoke_with_retry(self, chain: Any, inputs: Any) -> Any:
        """
        Invoke a chain, retrying the SAME model on transient Gemini errors
        (overload, rate limit, timeout) with increasing waits.  Other errors
        are raised immediately.
        """
        attempts = len(self._RETRY_WAITS) + 1
        for attempt in range(1, attempts + 1):
            try:
                return chain.invoke(inputs)
            except Exception as exc:
                transient = any(m.lower() in str(exc).lower() for m in self._TRANSIENT_MARKERS)
                if not transient or attempt == attempts:
                    raise
                wait = self._RETRY_WAITS[attempt - 1]
                self.logger.warning(
                    f"{self.name}: transient LLM error (attempt {attempt}/{attempts}); "
                    f"retrying in {wait}s."
                )
                time.sleep(wait)

    def execute(self, **kwargs: Any) -> AgentResult:
        """
        Run the agent's prompt through GEMINI and return an AgentResult.

        Default for PlannerAgent, FactCheckerAgent, CitationAgent.
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
            response = self._invoke_with_retry(chain, kwargs)
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

    def _run_gemini_chain(
        self,
        schema: type,
        prompt: Any,
        inputs: dict[str, Any],
    ) -> Any:
        """
        Execute a prompt through Gemini.
        Used by ResearcherAgent and any agent needing explicit Gemini routing.
        """
        self.logger.debug(f"{self.name}: _run_gemini_chain [Gemini] → {schema.__name__}")
        llm = get_structured_gemini_llm(schema)
        chain = prompt | llm
        return self._invoke_with_retry(chain, inputs)

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
        return self._invoke_with_retry(chain, inputs)