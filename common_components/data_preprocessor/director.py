from common_components.data_preprocessor.builder import PreprocessorBuilder

class PreprocessingDirector:
    """Director that defines the order of preprocessing steps."""

    def __init__(self, builder: PreprocessorBuilder):
        self.builder = builder

    def construct_builder(self):
        """Executes preprocessing steps for the chosen preprocessor."""
        self.builder.reset()
        self.builder.perform_preprocessing()

    def set_builder(self, builder: PreprocessorBuilder):
        """Allows changing the builder instance."""
        self.builder = builder
