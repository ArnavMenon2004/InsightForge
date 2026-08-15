from abc import ABC, abstractmethod


class BaseTool(ABC):
    """
    Base class for all tools.
    """

    @abstractmethod
    def execute(self, *args, **kwargs):
        """
        Execute the tool.
        """
        pass