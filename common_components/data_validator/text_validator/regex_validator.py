import re
from typing import Dict, Any

from common_components.validation_handler import ValidationHandler

class RegexValidator(ValidationHandler):
    """
    Concrete Handler that validates a field against a given regex pattern.
    """

    def __init__(self, field_name: str, pattern: str, error_message: str) -> None:
        """
        :param field_name: The field in the request dictionary to validate.
        :param pattern: The regex pattern to validate against.
        :param error_message: The custom error message if validation fails.
        """
        super().__init__()
        self.field_name = field_name
        self.pattern = re.compile(pattern)
        self.error_message = error_message

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that the specified field matches the regex pattern.
        """
        value = request.get(self.field_name)

        if not isinstance(value, str) or not self.pattern.search(value):
            return {"error": f"Validation failed: {self.error_message}"}

        if self._next_handler:
            return self._next_handler.validate(request)

        return {"success": True}