from abc import ABC, abstractmethod

class DataValidator(ABC):
    def __init__(self, next_validator=None):
        self.next_validator = next_validator

    def set_next(self, validator):
        """Sets the next validator in the chain."""
        self.next_validator = validator
        return validator  # Enables method chaining

    def validate(self, data):
        """Runs validation and passes data to the next validator if successful."""
        if not self._validate(data):
            return False  # Stop the chain if validation fails
        if self.next_validator:
            return self.next_validator.validate(data)
        return True  # If no more validators, return True

    @abstractmethod
    def _validate(self, data):
        """Each validator implements this method."""
        pass
