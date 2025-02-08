from typing import Dict, Any

from common_components.validation_handler import ValidationHandler

class LengthValidator(ValidationHandler):
    """
    Concrete Handler that ensures a string field's length is within the allowed range.
    """

    def __init__(self, field_name: str, min_length: int = 0, max_length: int = float('inf')) -> None:
        """
        :param field_name: The field in the request dictionary to validate.
        :param min_length: Minimum allowed length (default 0).
        :param max_length: Maximum allowed length (default infinity).
        """
        super().__init__()
        self.field_name = field_name
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks if the specified field's length is within the allowed range.
        If valid, passes the request to the next handler.
        """
        value = request.get(self.field_name)

        if not (self.min_length <= len(value) <= self.max_length):
            return {"error": f"Validation failed: '{self.field_name}' must be between {self.min_length} and {self.max_length} characters long"}

        # Pass request to the next handler if exists
        if self._next_handler:
            return self._next_handler.validate(request)

        return {"success": True}  # Final success if no further validation is required
