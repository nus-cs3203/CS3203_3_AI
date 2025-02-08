from typing import Dict, Any

from common_components.validation_handler import ValidationHandler

class OnlyStringValidator(ValidationHandler):
    """
    Concrete Handler that ensures a specific field contains only a string value.
    """

    def __init__(self, field_name: str) -> None:
        """
        :param field_name: The field in the request dictionary to validate.
        """
        super().__init__()
        self.field_name = field_name

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks if the specified field is of type string.
        If valid, passes the request to the next handler.
        """
        value = request.get(self.field_name)

        if not isinstance(value, str):
            return {"error": f"Validation failed: '{self.field_name}' must be a string"}

        # Pass request to the next handler if exists
        if self._next_handler:
            return self._next_handler.validate(request)

        return {"success": True}  # Final success if no further validation is required
