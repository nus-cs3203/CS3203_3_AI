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
