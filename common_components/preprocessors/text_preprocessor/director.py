from .builder import TextProcessorBuilder

class TextProcessorDirector:
    def __init__(self, builder: TextProcessorBuilder):
        self.builder = builder

    def construct_basic_pipeline(self):
        self.builder.add_tokenizer().add_normalizer().add_lemmatizer()
