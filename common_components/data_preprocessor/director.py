class PreprocessingDirector:
    """Director class that defines the order of preprocessing steps."""

    def __init__(self, builder):
        self.builder = builder

    def construct_general_preprocessing(self):
        """Constructs a general preprocessing pipeline."""
        return (
            self.builder
            .add_duplicate_remover()
            .add_missing_value_handler()
            .add_text_trimmer()
            .build()
        )

    def construct_text_preprocessing(self):
        """Constructs a text preprocessing pipeline."""
        return (
            self.builder
            .add_emoji_handler()
            .add_slang_handler()
            .add_text_trimmer()  
            .build()
        )

    def set_builder(self, builder):
        """Allows switching builders dynamically."""
        self.builder = builder
