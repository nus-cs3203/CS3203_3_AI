import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def lemmatize(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
