import re
from validators.base_validator import DataValidator

class URLValidator(DataValidator):
    URL_REGEX = re.compile(
        r"^(https?|ftp):\/\/"  # Protocol
        r"(([A-Za-z0-9-]+\.)+[A-Za-z]{2,6}|"  # Domain
        r"localhost|"  # Localhost
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # IPv4
        r"(:\d+)?(\/[^\s]*)?$"  # Port and path
    )

    def _validate(self, data):
        return bool(self.URL_REGEX.match(data))
