from enum import Enum

from pydantic import BaseModel, HttpUrl


class SourceType(str, Enum):
    OFFICIAL = "Official"
    ACADEMIC = "Academic"
    GOVERNMENT = "Government"
    NEWS = "News"
    DOCUMENTATION = "Documentation"
    WIKIPEDIA = "Wikipedia"
    BLOG = "Blog"
    OTHER = "Other"


class Source(BaseModel):
    """
    Normalized search result used throughout the application.
    """

    title: str

    url: HttpUrl

    snippet: str

    source_type: SourceType = SourceType.OTHER

    score: float | None = None