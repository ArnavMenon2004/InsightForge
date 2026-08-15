"""
WebFetchTool — fetches and extracts main text content from a URL.

This tool replaces the shallow Tavily snippets (typically 400–600 chars)
with real article content (1,500–3,000 chars of relevant text), which
gives the summarisation LLM substantially more material to work with.

Architecture note
-----------------
This is an application-level tool, not a provider.  It has no external
provider wrapper because HTTP is a standard library concern, not a
vendor-specific API.  ``httpx`` is used instead of ``requests`` for its
cleaner timeout API and better connection handling.

Dependencies: httpx, beautifulsoup4
Both are added to requirements.txt.
"""

from __future__ import annotations

import re

from utils.logger import logger

# Domains that reliably block scraping or return unhelpful content.
# Sources from these domains fall back to their Tavily snippet.
_BLOCKED_DOMAINS = frozenset(
    {
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "linkedin.com",
        "reddit.com",
        "youtube.com",
        "tiktok.com",
        "researchgate.net",  # requires login
        "jstor.org",          # paywalled
        "springer.com",       # paywalled
        "nature.com",         # paywalled
        "sciencedirect.com",  # paywalled
    }
)

# HTML tags whose content is never useful body text.
_NOISE_TAGS = {
    "script", "style", "nav", "footer", "header",
    "aside", "form", "button", "input", "noscript",
    "iframe", "svg", "figure", "figcaption",
}

_FETCH_TIMEOUT = 8  # seconds
_MIN_PARAGRAPH_WORDS = 15


class WebFetchTool:
    """
    Fetches a URL and extracts its main body text.

    The extractor is heuristic — it removes boilerplate (nav, scripts,
    ads), collects paragraphs longer than ``_MIN_PARAGRAPH_WORDS`` words,
    and returns up to ``max_chars`` characters of the result.

    Returns an empty string on any error so the caller can fall back
    to the Tavily snippet gracefully.
    """

    def fetch(self, url: str, max_chars: int = 3000) -> str:
        """
        Fetch ``url`` and return up to ``max_chars`` of extracted body text.

        Args:
            url:       The URL to fetch.
            max_chars: Maximum characters of body text to return.

        Returns:
            Extracted text, or empty string if fetch/parse fails.
        """
        if self._is_blocked(url):
            logger.debug(f"WebFetchTool: '{url}' is on blocked domain list.")
            return ""

        try:
            import httpx
        except ImportError:
            logger.warning(
                "WebFetchTool: httpx not installed. "
                "Run: pip install httpx beautifulsoup4"
            )
            return ""

        try:
            with httpx.Client(
                timeout=_FETCH_TIMEOUT,
                follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; InsightForge/1.0; "
                        "research assistant)"
                    )
                },
            ) as client:
                response = client.get(url)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type:
                logger.debug(
                    f"WebFetchTool: '{url}' returned non-HTML content-type."
                )
                return ""

            text = self._extract_text(response.text)
            result = text[:max_chars].strip()

            logger.debug(
                f"WebFetchTool: '{url}' → {len(result)} chars extracted."
            )
            return result

        except Exception as exc:
            logger.debug(f"WebFetchTool: Failed to fetch '{url}': {exc}")
            return ""

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_blocked(url: str) -> bool:
        """Return True if the URL's domain is on the blocked list."""
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().lstrip("www.")
        return any(blocked in domain for blocked in _BLOCKED_DOMAINS)

    @staticmethod
    def _extract_text(html: str) -> str:
        """
        Extract meaningful body text from raw HTML.

        Strategy:
        1. Parse with BeautifulSoup.
        2. Remove all noise tags (scripts, nav, footer, etc.).
        3. Collect paragraphs and list items with >= _MIN_PARAGRAPH_WORDS words.
        4. Join with newlines and collapse whitespace.

        Args:
            html: Raw HTML string.

        Returns:
            Cleaned plain-text string.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning(
                "WebFetchTool: beautifulsoup4 not installed. "
                "Run: pip install httpx beautifulsoup4"
            )
            return ""

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise tags in-place.
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

        paragraphs: list[str] = []

        for element in soup.find_all(["p", "li", "h2", "h3", "h4", "blockquote"]):
            text = element.get_text(separator=" ", strip=True)
            # Collapse internal whitespace.
            text = re.sub(r"\s+", " ", text).strip()
            if len(text.split()) >= _MIN_PARAGRAPH_WORDS:
                paragraphs.append(text)

        return "\n\n".join(paragraphs)