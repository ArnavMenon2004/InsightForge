from dotenv import load_dotenv
import os

from utils.logger import logger

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GROQ_API_KEY:
    logger.info("Groq API key found (optional).")
else:
    logger.info("Groq API key not set — all agents routing through Gemini.")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found.")
    raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")

logger.info("Environment variables loaded successfully.")

# ---------------------------------------------------------------------------
# Multi-provider LLM routing
#
# InsightForge uses two API providers to maximise throughput on free tiers:
#
#   GROQ  (llama-3.3-70b-versatile)  — WriterAgent only
#     Free tier: 12,000 TPM / 100,000 TPD
#     Used for writing because llama-3.3-70b produces the best prose quality.
#     ~7,266 tokens per run (writer only).
#
#   GEMINI (gemini-2.0-flash)  — all other agents
#     Free tier: 1,000,000 TPM / 1,500 requests per day
#     Used for planning, research, fact-checking, citation.
#     Effectively no TPM limit — ideal for high-volume calls.
#     ~16,710 tokens per run routed here instead of Groq.
#
# Result: Groq daily budget drops from ~24,000 → ~7,266 tokens per run,
# giving ~13 full runs per day on Groq instead of ~4.
# ---------------------------------------------------------------------------

# Pause between consecutive LLM calls (researcher and writer loops).
# Can be reduced now that researcher uses Gemini (1M TPM).
SECTION_RATE_LIMIT_DELAY: int = 5

# --- Researcher budget (Gemini Flash, 1M TPM — generous limits) ---
MAX_RESULTS_PER_QUERY: int = 3
MAX_SOURCES_FOR_PROMPT: int = 5
SNIPPET_CHAR_LIMIT: int = 1000  # chars of fetched page content per source

# --- Source quality filter ---
MIN_SOURCE_SCORE: float = 0.5

# --- FactChecker / CitationAgent budget (Gemini Flash) ---
# Gemini handles larger contexts well — increased caps vs Groq limits.
FC_MAX_SOURCES: int = 15
FC_SNIPPET_CHARS: int = 400