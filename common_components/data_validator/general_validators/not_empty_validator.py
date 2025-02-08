import math
from typing import Optional, Dict, Any

from common_components.validation_handler import ValidationHandler

class NotEmptyValidator(ValidationHandler):
    """
    Concrete Handler that validates if a specific field in the request is neither None nor NaN.
    """

    def __init__(self, field_name: str) -> None:
        """
        :param field_name: The field in the request dictionary to validate.
        """
        super().__init__()
        self.field_name = field_name

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks if the specified field is not None or NaN.
        If valid, passes the request to the next handler.
        """
        value = request.get(self.field_name)

        if value is None or (isinstance(value, float) and math.isnan(value)):
            return {"error": f"Validation failed: '{self.field_name}' cannot be null or NaN"}

        # Pass request to the next handler in the chain if exists
        if self._next_handler:
            return self._next_handler.validate(request)

        return {"success": True}  # Final success if no further validation is required
