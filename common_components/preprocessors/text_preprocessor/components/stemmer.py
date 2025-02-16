from nltk.stem import PorterStemmer

class Stemmer:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def process(self, words):
        return [self.stemmer.stem(word) for word in words]
