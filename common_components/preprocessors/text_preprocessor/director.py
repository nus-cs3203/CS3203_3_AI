from builder import TextProcessorBuilderInterface
from components.tokenizer import Tokenizer
from components.normalizer import Normalizer
from components.lemmatizer import Lemmatizer

class TextProcessorDirector:
    def __init__(self, builder: TextProcessorBuilderInterface):
        self.builder = builder
    
    def construct_basic_pipeline(self):
        self.builder.set_tokenizer(Tokenizer())
        self.builder.set_normalizer(Normalizer())
        self.builder.set_lemmatizer(Lemmatizer())
    
    def construct_advanced_pipeline(self):
        self.builder.set_tokenizer(Tokenizer())
        self.builder.set_normalizer(Normalizer())
        self.builder.set_lemmatizer(Lemmatizer())  # Extend with more steps if needed
