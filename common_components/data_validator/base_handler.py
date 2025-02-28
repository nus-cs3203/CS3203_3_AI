from common_components.data_validator.validation_handler import ValidationHandler
from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd

class BaseValidationHandler(ValidationHandler):
    """
    The BaseValidationHandler class provides the chain-handling mechanism.
    """
    def __init__(self) -> None:
        self._next_handler: Optional['ValidationHandler'] = None

    def set_next(self, handler: 'ValidationHandler') -> 'ValidationHandler':
        """
        Set the next handler in the chain and return it to allow chaining.
        """
        self._next_handler = handler
        return handler  

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Perform validation and pass to the next handler if available.
        """
        return self._validate_next(df)

    def _validate_next(self, df: pd.DataFrame) -> dict:
        """
        Pass the DataFrame to the next handler in the chain, if any.
        If there is no next handler, return success.
        """
        if self._next_handler:
            return self._next_handler.validate(df)
        return {"success": True}
