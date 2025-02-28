from common_components.data_preprocessor.builder import PreprocessorBuilder

class PreprocessingDirector:
    """Director that defines the order of preprocessing steps."""

    def __init__(self, builder: PreprocessorBuilder):
        self.builder = builder

    def construct_general_builder(self):
        """Executes preprocessing steps for the general preprocessor."""
        self.builder.remove_duplicates()
        self.builder.handle_missing_values()
        self.builder.normalize_text()
        self.builder.handle_slang_and_emojis()
        self.builder.lemmatize()
        self.builder.remove_stopwords()
        self.builder.stem_words()

    def set_builder(self, builder: PreprocessorBuilder):
        """Allows changing the builder instance."""
        self.builder = builder
