from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class ValidationHandler(ABC):
    """
    The Validation Handler interface declares a method for building the chain of validator handlers.
    It also declares a method for executing a request on a pandas DataFrame.
    """

    def __init__(self) -> None:
        self._next_handler: Optional['ValidationHandler'] = None

    def set_next(self, handler: 'ValidationHandler') -> 'ValidationHandler':
        """
        Set the next handler in the chain.
        Returns the handler to allow chaining: handler1.set_next(handler2).set_next(handler3)
        """
        self._next_handler = handler
        return handler  

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> dict:
        """
        Perform validation and optionally pass to the next handler.
        Concrete classes must implement this method.
        """
        pass

    def _validate_next(self, df: pd.DataFrame) -> dict:
        """
        Pass the DataFrame to the next handler in the chain, if any.
        If there is no next handler, return success.
        """
        if self._next_handler:
            return self._next_handler.validate(df)
        return {"success": True}
