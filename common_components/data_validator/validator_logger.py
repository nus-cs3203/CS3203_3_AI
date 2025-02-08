import logging

class ValidatorLogger:
    """
    A separate logger component for logging validation attempts and results.
    """

    def __init__(self, log_level=logging.INFO):
        """
        Initialize the logger with a given log level.
        """
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("ValidatorLogger")

    def log_success(self, field_name: str):
        """Log a successful validation."""
        self.logger.info(f"Validation passed for field: {field_name}")

    def log_failure(self, field_name: str, error_message: str):
        """Log a failed validation."""
        self.logger.warning(f"Validation failed for field '{field_name}': {error_message}")

    def log_request(self, request: dict):
        """Log the incoming request."""
        self.logger.debug(f"Processing request: {request}")
