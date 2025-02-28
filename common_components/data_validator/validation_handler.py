from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class ValidationHandler(ABC):
    """
    The ValidationHandler interface declares methods for executing a validation request on a pandas DataFrame
    and setting the next handler in the chain.
    """
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> dict:
        """
        Perform validation and optionally pass to the next handler.
        Concrete classes must implement this method.
        """
        pass

    @abstractmethod
    def set_next(self, handler: 'ValidationHandler') -> 'ValidationHandler':
        """
        Set the next handler in the chain and return it to allow chaining.
        """
        pass