import re
from validators.base_validator import DataValidator

class RegexValidator(DataValidator):
    def __init__(self, pattern, next_validator=None):
        super().__init__(next_validator)
        self.pattern = pattern

    def _validate(self, data):
        return bool(re.match(self.pattern, data))
