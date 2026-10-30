from __future__ import annotations

from urllib.parse import urlparse

from config.settings import LOW_QUALITY_DOMAINS, MIN_SOURCE_SCORE
from providers.tavily_provider import TavilyProvider
from schemas.source import Source, SourceType
from tools.base_tool import BaseTool
from utils.logger import logger


class WebSearchTool(BaseTool):
    """
    Application-level web search tool.

    Wraps ``TavilyProvider`` and normalises every raw result into a
    typed ``Source`` object.  Applies a relevance score filter
    (``MIN_SOURCE_SCORE``) to discard SEO spam and low-quality pages
    before they reach the researcher.

    Args:
        provider: Optional provider instance.  Defaults to a freshly
            constructed ``TavilyProvider``.  Pass a mock here in tests.
    """

    def __init__(self, provider: TavilyProvider | None = None) -> None:
        self._provider = provider or TavilyProvider()

    def execute(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[Source]:
        """
        Run a web search and return quality-filtered, normalised Sources.

        Results with a Tavily relevance score below ``MIN_SOURCE_SCORE``
        are discarded.  Results that cannot be parsed are skipped and
        logged rather than raising an exception.

        Args:
            query:       The search query string.
            max_results: Maximum number of results to request.

        Returns:
            A list of ``Source`` objects (may be empty).
        """
        logger.debug(f"WebSearchTool: searching → '{query}'")

        # Ask for extra results: some are dropped by the filters below.
        response = self._provider.search(
            query=query,
            max_results=min(max_results * 2, 20),
        )

        sources: list[Source] = []
        discarded = 0

        for raw in response.get("results", []):
            source = self._parse_result(raw)
            if source is None:
                continue

            if self._is_low_quality_domain(str(source.url)):
                logger.debug(f"WebSearchTool: Discarding low-quality domain ({source.url}).")
                discarded += 1
                continue

            # Quality filter — discard low-relevance results.
            if source.score is not None and source.score < MIN_SOURCE_SCORE:
                logger.debug(
                    f"WebSearchTool: Discarding low-quality source "
                    f"(score={source.score:.2f}, title='{source.title}')."
                )
                discarded += 1
                continue

            sources.append(source)

        sources = sources[:max_results]

        logger.debug(
            f"WebSearchTool: '{query}' → {len(sources)} sources kept, "
            f"{discarded} discarded (score < {MIN_SOURCE_SCORE})."
        )

        return sources

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_low_quality_domain(url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in LOW_QUALITY_DOMAINS)

    @staticmethod
    def _parse_result(raw: dict) -> Source | None:
        """
        Convert a single raw Tavily result dict into a ``Source``.

        Returns ``None`` and logs a warning if the result is malformed.
        """
        url = raw.get("url", "")
        title = raw.get("title", "")
        snippet = raw.get("content", "")

        if not url:
            logger.warning(
                f"WebSearchTool: Skipping result with missing URL "
                f"(title='{title}')."
            )
            return None

        try:
            return Source(
                title=title,
                url=url,
                snippet=snippet,
                source_type=SourceType.OTHER,
                score=raw.get("score"),
            )
        except Exception as exc:
            logger.warning(
                f"WebSearchTool: Could not parse result "
                f"(url='{url}'): {exc}"
            )
            return None