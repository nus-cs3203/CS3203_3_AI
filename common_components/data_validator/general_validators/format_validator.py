import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Callable

from common_components.validation_handler import ValidationHandler

class FormatValidator(ValidationHandler):
    """
    Concrete Handler that validates if a field matches an allowed format.
    Supported formats: Pandas DataFrame, JSON (dict), NumPy array, or .txt file.
    """

    FORMAT_CHECKS = {
        "pandas": lambda value: isinstance(value, pd.DataFrame),
        "json": lambda value: isinstance(value, dict) and FormatValidator.is_json_serializable(value),
        "numpy": lambda value: isinstance(value, np.ndarray),
        "txt": lambda value: isinstance(value, str) and value.lower().endswith('.txt')
    }

    def __init__(self, field_name: str, allowed_formats: list) -> None:
        """
        :param field_name: The field in the request dictionary to validate.
        :param allowed_formats: List of allowed formats (e.g., ['pandas', 'json', 'numpy', 'txt']).
        """
        super().__init__()
        self.field_name = field_name
        self.allowed_formats = set(allowed_formats)

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that the specified field matches one of the allowed formats.
        """
        value = request.get(self.field_name)

        # Check if the value matches any allowed format
        if any(check(value) for fmt, check in self.FORMAT_CHECKS.items() if fmt in self.allowed_formats):
            return self._validate_next(request)

        return {"error": f"Validation failed: '{self.field_name}' must be one of {self.allowed_formats}"}

    def _validate_next(self, request):
        """ Pass the request to the next handler if exists, else return success """
        if self._next_handler:
            return self._next_handler.validate(request)
        return {"success": True}

    @staticmethod
    def is_json_serializable(value) -> bool:
        """ Helper function to check if a dictionary is JSON serializable """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
