import numpy as np
from gensim.models import Word2Vec

class TextEncoder:
    def __init__(self, model_path=None):
        if model_path:
            self.model = Word2Vec.load(model_path)  # Load a pre-trained model
        else:
            self.model = None

    def process(self, words):
        if self.model:
            return [self.model.wv[word] if word in self.model.wv else np.zeros(300) for word in words]
        else:
            return words  # Return words as is if no model is loaded
