from abc import ABC, abstractmethod
from components.tokenizer import Tokenizer
from components.normalizer import Normalizer
from components.lemmatizer import Lemmatizer

class TextProcessorBuilderInterface(ABC):
    
    @abstractmethod
    def set_tokenizer(self, tokenizer: Tokenizer):
        pass
    
    @abstractmethod
    def set_normalizer(self, normalizer: Normalizer):
        pass
    
    @abstractmethod
    def set_lemmatizer(self, lemmatizer: Lemmatizer):
        pass
    
    @abstractmethod
    def get_tokenizer(self) -> Tokenizer:
        pass
    
    @abstractmethod
    def get_normalizer(self) -> Normalizer:
        pass
    
    @abstractmethod
    def get_lemmatizer(self) -> Lemmatizer:
        pass
