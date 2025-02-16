import pandas as pd
import numpy as np
from typing import Dict, Any
from common_components.data_validator.validation_handler import ValidationHandler
from common_components.data_validator.validator_logger import ValidatorLogger

class NotEmptyValidator(ValidationHandler):
    """
    Validates that a given field is not None, empty, or NaN.
    """

    def __init__(self, field_name: str, logger: ValidatorLogger) -> None:
        super().__init__()
        self.field_name = field_name
        self.logger = logger

    def validate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.log_request(request)
        value = request.get(self.field_name)

        # Check for None, empty string, or NaN
        if value is None or (isinstance(value, str) and value.strip() == "") or (isinstance(value, (float, int)) and np.isnan(value)):
            error_message = f"Validation failed: '{self.field_name}' cannot be null, empty, or NaN."
            self.logger.log_failure(self.field_name, error_message)
            return {"error": error_message}

        self.logger.log_success(self.field_name)
        return self._validate_next(request)
