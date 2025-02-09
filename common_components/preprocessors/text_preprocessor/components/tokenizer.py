import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class Tokenizer:
    def tokenize(self, text):
        return word_tokenize(text)
