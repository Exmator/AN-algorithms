from abc import ABC, abstractmethod

class Parameters(ABC):
      """Generic entry parameters"""

class Result(ABC):
    """Generic algorithm result"""

    @abstractmethod
    def __str__(self) -> str:
        """Human-readable summary of the result"""

class Algorithm(ABC):
    @abstractmethod
    def _validate(self, p:Parameters) -> None:
        """Validate entry parameters"""

    @abstractmethod
    def _execute(self, p:Parameters) -> Result:
        """Execute the algorithm"""

    def run(self, p:Parameters) -> Result:
        self._validate(p)
        return self._execute(p)