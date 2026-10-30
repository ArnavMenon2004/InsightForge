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

# Domains dropped from search results: social posts, forums, blogs and press-release
# wires are weak evidence for a research report. Edit freely.
LOW_QUALITY_DOMAINS: tuple[str, ...] = (
    "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com",
    "reddit.com", "quora.com", "medium.com", "youtube.com", "youtu.be",
    "pinterest.com", "linkedin.com", "finance.yahoo.com", "wikipedia.org",
    "cryptorank.io", "cointelegraph.com", "coindesk.com", "decrypt.co", "bitcoinist.com",
    "newsbtc.com", "stocktitan.net", "benzinga.com", "seekingalpha.com", "investing.com",
    "marketbeat.com", "tipranks.com", "zacks.com", "fool.com",
)

# --- FactChecker / CitationAgent budget ---
# Must cover every source the researcher collects, otherwise claims backed by
# later sources are wrongly marked UNVERIFIABLE (a run can gather 50+).
FC_MAX_SOURCES: int = 80
# Same as SNIPPET_CHAR_LIMIT: the checker must see what the writer saw, or it
# marks claims from later in a page as UNVERIFIABLE.
FC_SNIPPET_CHARS: int = SNIPPET_CHAR_LIMIT
# Paper to PPT: max characters of paper text sent to the LLM (~10k tokens)
PPT_PAPER_CHAR_LIMIT: int = 40000
