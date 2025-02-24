import pandas as pd
from common_components.data_preprocessor.builder import PreprocessorBuilder


class PreprocessingDirector:
    """Director that defines the order of preprocessing steps."""

    def __init__(self, builder: PreprocessorBuilder):
        self.builder = builder

    def construct(self, data: pd.DataFrame):
        """Executes preprocessing steps in order."""
        self.builder.load_data(data)
        self.builder.remove_duplicates()
        self.builder.handle_missing_values()
        self.builder.normalize_text()
        self.builder.handle_slang_and_emojis()
        self.builder.remove_stopwords()
        self.builder.stem_words()
        return self.builder.get_result()

    def set_builder(self, builder: PreprocessorBuilder):
        """Allows switching builders dynamically."""
        self.builder = builder
