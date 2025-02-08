from typing import Dict, Any
from common_components.data_validator.validation_handler import ValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class NumericRangeValidator(ValidationHandler):
    """
    Validates that a number is within a specified range.
    """

    def __init__(self, field_name: str, min_value: float, max_value: float, logger: ValidatorLogger) -> None:
        super().__init__()
        self.field_name = field_name
        self.min_value = min_value
        self.max_value = max_value
        self.logger = logger

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_request(request)
        value = request.get(self.field_name)

        if not isinstance(value, (int, float)) or not (self.min_value <= value <= self.max_value):
            error_message = f"Validation failed: '{self.field_name}' must be between {self.min_value} and {self.max_value}."
            self.logger.log_failure(self.field_name, error_message)
            return {"error": error_message}

        self.logger.log_success(self.field_name)
        return self._validate_next(request)
