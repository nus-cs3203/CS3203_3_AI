from abc import ABC, abstractmethod

class Preprocessor(ABC):
    """Abstract Preprocessor defining the template method."""

    def preprocess(self, data):
        """Template method defining the preprocessing workflow."""
        self.load_data(data)
        processed_data = self.process_data()
        return self.finalize(processed_data)

    @abstractmethod
    def load_data(self, data):
        pass

    @abstractmethod
    def process_data(self):
        pass

    def finalize(self, data):
        """Optional step to finalize output (can be overridden)."""
        return data
