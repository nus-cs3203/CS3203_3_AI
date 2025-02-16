from components.tokenizer import Tokenizer
from components.normalizer import Normalizer
from components.lemmatizer import Lemmatizer

class TextProcessorBuilder:
    def __init__(self):
        self.steps = []

    def add_tokenizer(self):
        self.steps.append(Tokenizer())
        return self

    def add_normalizer(self):
        self.steps.append(Normalizer())
        return self

    def add_lemmatizer(self):
        self.steps.append(Lemmatizer())
        return self

    def get_steps(self):
        return self.steps
