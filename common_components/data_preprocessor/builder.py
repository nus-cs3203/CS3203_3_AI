from abc import ABC, abstractmethod

class PreprocessorBuilder(ABC):
    """
    Interface for building a preprocessing pipeline.
    """
    
    @abstractmethod
    def reset(self):
        """Resets the builder to its initial state."""
        pass

    @abstractmethod
    def perform_preprocessing(self):
        """Executes the preprocessing steps."""
        pass

    @abstractmethod
    def get_result(self):
        """Returns the processed data."""
        pass
