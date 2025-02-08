from typing import Dict, Any
from common_components.validation_handler import ValidationHandler
from validator_logger import ValidatorLogger

class OnlyStringValidator(ValidationHandler):
    """
    Validates that a given field is a string.
    """

    def __init__(self, field_name: str, logger: ValidatorLogger) -> None:
        super().__init__()
        self.field_name = field_name
        self.logger = logger

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_request(request)
        value = request.get(self.field_name)

        if not isinstance(value, str):
            error_message = f"Validation failed: '{self.field_name}' must be a string."
            self.logger.log_failure(self.field_name, error_message)
            return {"error": error_message}

        self.logger.log_success(self.field_name)
        return self._validate_next(request)
