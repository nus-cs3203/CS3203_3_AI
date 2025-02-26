import logging

class ValidatorLogger:
    """
    A separate logger component for logging validation attempts and results.
    """

    def __init__(self, log_level=logging.INFO):
        """
        Initialize the logger with a given log level.
        """
        self.logger = logging.getLogger("ValidatorLogger")
        self.logger.disabled = True  # 完全禁用日志输出

    def log_success(self, field_name: str):
        """Log a successful validation."""
        pass  # 不执行任何日志记录

    def log_failure(self, field_name: str, error_message: str):
        """Log a failed validation."""
        pass  # 不执行任何日志记录

    def log_request(self, request: dict):
        """Log the incoming request."""
        pass  # 不执行任何日志记录
