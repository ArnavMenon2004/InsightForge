from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class AgentResult(BaseModel, Generic[T]):
    """
    Standard response returned by every agent.
    """

    success: bool
    data: T
    model: str
    execution_time: float