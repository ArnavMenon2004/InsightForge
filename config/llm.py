from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import GEMINI_API_KEY
from utils.logger import logger

# All agents run on Gemini. Change the model here and nowhere else.
GEMINI_MODEL = "models/gemini-3.1-flash-lite"

# Legacy alias — used in AgentResult.model field for logging
SUMMARISER_MODEL = GEMINI_MODEL

_gemini_llm: ChatGoogleGenerativeAI | None = None


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """
    Return the shared Gemini LLM instance.
    Lazy-initialised on first call.
    """
    global _gemini_llm

    if _gemini_llm is None:
        logger.info(f"Initializing Gemini LLM ({GEMINI_MODEL})...")
        _gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
            # Fail fast; BaseAgent._invoke_with_retry handles transient
            # overloads (503/429) with longer, logged waits.
            max_retries=2,
            timeout=120,
        )
        logger.info("Gemini LLM initialized.")

    return _gemini_llm


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------

def get_structured_gemini_llm(schema: type) -> ChatGoogleGenerativeAI:
    """
    Return Gemini LLM bound to a structured output schema.
    Used by PlannerAgent, ResearcherAgent, FactCheckerAgent, CitationAgent.
    """
    return get_gemini_llm().with_structured_output(schema)


# Keep legacy name so ResearcherAgent._run_summariser_chain() still works
# without changes — it calls get_structured_summariser_llm which now
# routes to Gemini.
def get_structured_summariser_llm(schema: type) -> ChatGoogleGenerativeAI:
    """
    Return the summariser LLM bound to a schema (Gemini).
    Called by ResearcherAgent._run_summariser_chain().
    """
    return get_structured_gemini_llm(schema)