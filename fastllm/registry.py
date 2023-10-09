from abc import ABC, abstractmethod
from typing import Callable


class Registry(ABC):
    @abstractmethod
    def register(self, name: str, description: str, func: Callable):
        pass

    # @abstractmethod
    def get(self, name: str):
        pass

    @abstractmethod
    def list(self):
        pass
