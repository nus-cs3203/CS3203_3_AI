from validators.base_validator import DataValidator

class MinLengthValidator(DataValidator):
    def __init__(self, min_length, next_validator=None):
        super().__init__(next_validator)
        self.min_length = min_length

    def _validate(self, data):
        return len(data) >= self.min_length if isinstance(data, str) else False
