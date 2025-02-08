from typing import Dict, Any
from common_components.validation_handler import ValidationHandler
from validator_logger import ValidatorLogger

class LengthValidator(ValidationHandler):
    """
    Validates that a given string field has a length within the specified range.
    """

    def __init__(self, field_name: str, min_length: int, max_length: int, logger: ValidatorLogger) -> None:
        super().__init__()
        self.field_name = field_name
        self.min_length = min_length
        self.max_length = max_length
        self.logger = logger

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_request(request)
        value = request.get(self.field_name)

        if not isinstance(value, str) or not (self.min_length <= len(value) <= self.max_length):
            error_message = f"Validation failed: '{self.field_name}' length must be between {self.min_length} and {self.max_length} characters."
            self.logger.log_failure(self.field_name, error_message)
            return {"error": error_message}

        self.logger.log_success(self.field_name)
        return self._validate_next(request)