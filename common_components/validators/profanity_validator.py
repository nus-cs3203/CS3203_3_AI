from validators.base_validator import DataValidator

class ProfanityFilterValidator(DataValidator):
    PROFANITY_LIST = {"badword1", "badword2"}  # Replace with real words

    def _validate(self, data):
        return not any(word in data.lower() for word in self.PROFANITY_LIST)
