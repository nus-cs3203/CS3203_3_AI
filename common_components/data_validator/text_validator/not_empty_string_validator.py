from typing import Dict, Any

from common_components.validation_handler import ValidationHandler

class NonEmptyStringValidator(ValidationHandler):
    """
    Concrete Handler that validates whether a specific field is a non-empty string
    (not empty and not just whitespace).
    """

    def __init__(self, field_name: str) -> None:
        """
        :param field_name: The field in the request dictionary to validate.
        """
        super().__init__()
        self.field_name = field_name

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates that the specified field is a non-empty string.
        """
        value = request.get(self.field_name)

        # Check if the value is a string and is not empty or just whitespace
        if not isinstance(value, str) or value.strip() == "":
            return {"error": f"Validation failed: '{self.field_name}' must be a non-empty string"}

        # Pass to the next handler in the chain if it exists
        if self._next_handler:
            return self._next_handler.validate(request)

        return {"success": True}
