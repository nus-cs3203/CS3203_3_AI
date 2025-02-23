from abc import ABC, abstractmethod
import pandas as pd

class PreprocessorBuilder(ABC):
    """Interface for Preprocessor Builder defining common preprocessing steps."""

    def __init__(self):
        self.data = None  # Store the DataFrame

    @abstractmethod
    def load_data(self, data: pd.DataFrame):
        """Load DataFrame into the pipeline."""
        pass

    @abstractmethod
    def process_data(self):
        """Apply processing steps (e.g., removing duplicates, text processing)."""
        pass

    @abstractmethod
    def validate_data(self):
        """Validate processed data (e.g., checking for missing critical values)."""
        pass

    @abstractmethod
    def get_result(self) -> pd.DataFrame:
        """Return the final preprocessed DataFrame."""
        pass
