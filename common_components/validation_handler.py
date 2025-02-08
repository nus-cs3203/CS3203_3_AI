from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class ValidationHandler(ABC):
    """
    The Validation Handler interface declares a method for building the chain of validator handlers.
    It also declares a method for executing a request.
    """

    def __init__(self) -> None:
        self._next_handler: Optional['ValidationHandler'] = None

    def set_next(self, handler: 'ValidationHandler') -> 'ValidationHandler':
        """Set the next handler in the chain."""
        self._next_handler = handler
        return handler  # Allows chaining: handler1.set_next(handler2).set_next(handler3)

    @abstractmethod
    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform validation and optionally pass to the next handler."""
        pass
