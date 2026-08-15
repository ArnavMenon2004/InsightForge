import os
from config.settings import TAVILY_API_KEY
from tavily import TavilyClient


class TavilyProvider:

    def __init__(self):

        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is missing.")

        self.client = TavilyClient(api_key=TAVILY_API_KEY)

    def search(
        self,
        query: str,
        max_results: int = 5,
    ):

        return self.client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
            include_raw_content=False,
        )