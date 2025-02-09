from builder import TextProcessorBuilderInterface

class TextProcessorComponent:
    def __init__(self, builder: TextProcessorBuilderInterface):
        self.tokenizer = builder.get_tokenizer()
        self.normalizer = builder.get_normalizer()
        self.lemmatizer = builder.get_lemmatizer()
    
    def process(self, text):
        text = self.normalizer.normalize(text)
        tokens = self.tokenizer.tokenize(text)
        lemmatized_tokens = self.lemmatizer.lemmatize(tokens)
        return lemmatized_tokens
