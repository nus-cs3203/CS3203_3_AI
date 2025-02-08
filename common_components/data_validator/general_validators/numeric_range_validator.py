from typing import Dict, Any

from common_components.validation_handler import ValidationHandler

class RangeValidator(ValidationHandler):
    """
    Concrete Handler that ensures a numeric field is within a specified range.
    """

    def __init__(self, field_name: str, min_value: float = float('-inf'), max_value: float = float('inf')) -> None:
        """
        :param field_name: The field in the request dictionary to validate.
        :param min_value: The minimum acceptable value (inclusive).
        :param max_value: The maximum acceptable value (inclusive).
        """
        super().__init__()
        self.field_name = field_name
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that the specified field is a number within the range.
        """
        value = request.get(self.field_name)

        try:
            num = float(value)  # Convert to float to handle both int and float
        except (ValueError, TypeError):
            return {"error": f"Validation failed: '{self.field_name}' must be a number"}

        if not (self.min_value <= num <= self.max_value):
            return {"error": f"Validation failed: '{self.field_name}' must be between {self.min_value} and {self.max_value}"}

        if self._next_handler:
            return self._next_handler.validate(request)

        return {"success": True}
