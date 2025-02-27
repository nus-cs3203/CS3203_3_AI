from abc import ABC, abstractmethod
import pandas as pd

class PreprocessorBuilder(ABC):
    """Abstract Builder defining preprocessing steps."""

    def __init__(self):
        self.data = None  # Store the DataFrame

    def load_data(self, data: pd.DataFrame):
        """Load DataFrame into the pipeline."""
        self.data = data

    @abstractmethod
    def remove_duplicates(self):
        """Removes duplicate entries."""
        pass

    @abstractmethod
    def handle_missing_values(self):
        """Handles missing values in critical columns."""
        pass

    @abstractmethod
    def normalize_text(self):
        """Normalizes text (lowercasing, trimming, etc.)."""
        pass

    @abstractmethod
    def handle_slang_and_emojis(self):
        """Handles slangs and emojis in text columns."""
        pass

    @abstractmethod
    def lemmatize(self):
        """Applies lemmatization."""
        pass

    @abstractmethod
    def remove_stopwords(self):
        """Removes stopwords."""
        pass

    @abstractmethod
    def stem_words(self):
        """Applies stemming."""
        pass

    def get_result(self) -> pd.DataFrame:
        """Returns the preprocessed DataFrame."""
        return self.data
