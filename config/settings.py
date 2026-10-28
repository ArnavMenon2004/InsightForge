from dotenv import load_dotenv
import os

from utils.logger import logger

load_dotenv()

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found.")
    raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")

logger.info("Environment variables loaded successfully.")

# ---------------------------------------------------------------------------
# LLM routing
#
# Every agent runs on Gemini (see config/llm.py for the model).
# ---------------------------------------------------------------------------

# Pause between consecutive LLM calls (researcher and writer loops).
SECTION_RATE_LIMIT_DELAY: int = 5

# --- Researcher budget ---
MAX_RESULTS_PER_QUERY: int = 3
MAX_SOURCES_FOR_PROMPT: int = 5
SNIPPET_CHAR_LIMIT: int = 1000  # chars of fetched page content per source

# --- Source quality filter ---
MIN_SOURCE_SCORE: float = 0.5

# --- FactChecker / CitationAgent budget ---
FC_MAX_SOURCES: int = 15
FC_SNIPPET_CHARS: int = 400
# Paper to PPT: max characters of paper text sent to the LLM (~10k tokens)
PPT_PAPER_CHAR_LIMIT: int = 40000
