from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import GROQ_API_KEY, GEMINI_API_KEY
from utils.logger import logger

# ---------------------------------------------------------------------------
# Multi-provider LLM routing
#
# Provider assignments:
#
#   GROQ — llama-3.3-70b-versatile
#     WriterAgent only. Best prose quality. 12,000 TPM / 100,000 TPD.
#
#   GEMINI — gemini-2.0-flash
#     PlannerAgent, ResearcherAgent, FactCheckerAgent, CitationAgent.
#     1,000,000 TPM — effectively no per-minute limit on free tier.
#     Reliable structured output support.
#
# This routing reduces Groq usage from ~24,000 → ~7,266 tokens per run,
# giving ~13 full runs per day instead of ~4.
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "models/gemini-3.1-flash-lite"

# Legacy aliases — used in AgentResult.model field for logging
PRIMARY_MODEL = GROQ_MODEL
SUMMARISER_MODEL = GEMINI_MODEL

_groq_llm: ChatGroq | None = None
_gemini_llm: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGroq:
    """
    Return the Groq LLM instance (llama-3.3-70b-versatile).
    Used exclusively by WriterAgent for highest prose quality.
    Lazy-initialised on first call.
    """
    global _groq_llm

    if _groq_llm is None:
        logger.info(f"Initializing Groq LLM ({GROQ_MODEL})...")
        _groq_llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.2,
        )
        logger.info("Groq LLM initialized.")

    return _groq_llm


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    """
    Return the Gemini LLM instance (gemini-2.0-flash).
    Used by PlannerAgent, ResearcherAgent, FactCheckerAgent, CitationAgent.
    1,000,000 TPM free tier — no per-minute bottleneck.
    Lazy-initialised on first call.
    """
    global _gemini_llm

    if _gemini_llm is None:
        logger.info(f"Initializing Gemini LLM ({GEMINI_MODEL})...")
        _gemini_llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.2,
        )
        logger.info("Gemini LLM initialized.")

    return _gemini_llm


# ---------------------------------------------------------------------------
# Structured output helpers
# ---------------------------------------------------------------------------

def get_structured_llm(schema: type) -> ChatGroq:
    """
    Return Groq LLM bound to a structured output schema.
    Used by WriterAgent via BaseAgent.execute().
    """
    return get_llm().with_structured_output(schema)


def get_structured_gemini_llm(schema: type) -> ChatGoogleGenerativeAI:
    """
    Return Gemini LLM bound to a structured output schema.
    Used by PlannerAgent, ResearcherAgent, FactCheckerAgent, CitationAgent.
    """
    return get_gemini_llm().with_structured_output(schema)


# Keep legacy name so ResearcherAgent._run_summariser_chain() still works
# without changes — it calls get_structured_summariser_llm which now
# routes to Gemini instead of Groq.
def get_structured_summariser_llm(schema: type) -> ChatGoogleGenerativeAI:
    """
    Return the summariser LLM bound to a schema (now Gemini).
    Called by ResearcherAgent._run_summariser_chain().
    """
    return get_structured_gemini_llm(schema)