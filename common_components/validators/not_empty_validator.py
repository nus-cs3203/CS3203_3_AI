from validators.base_validator import DataValidator

class NotEmptyValidator(DataValidator):
    def _validate(self, data):
        return bool(data.strip()) if isinstance(data, str) else False
