import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

class StopwordRemover:
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))

    def process(self, text):
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return " ".join(filtered_words)
