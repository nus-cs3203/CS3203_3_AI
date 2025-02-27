
from common_components.data_preprocessor.builder import PreprocessorBuilder
from common_components.data_preprocessor.components.duplicate_remover import DuplicateRemover
from common_components.data_preprocessor.components.emoji_slang_handler import EmojiSlangHandler
from common_components.data_preprocessor.components.lemmatizer import Lemmatizer
from common_components.data_preprocessor.components.missing_values_handler import MissingValueHandler
from common_components.data_preprocessor.components.normalizer import Normalizer
from common_components.data_preprocessor.components.stemmer import Stemmer
from common_components.data_preprocessor.components.stopword_remover import StopwordRemover
from common_components.data_preprocessor.components.text_trimmer import TextTrimmer
from common_components.data_preprocessor.components.tokenizer import Tokenizer



class GeneralPreprocessorBuilder(PreprocessorBuilder):
    """Concrete Builder implementing preprocessing steps."""

    def __init__(self, critical_columns, text_columns):
        super().__init__()
        self.critical_columns = critical_columns
        self.text_columns = text_columns
        self.duplicate_remover = DuplicateRemover()
        self.missing_value_handler = MissingValueHandler(critical_columns)
        self.text_trimmer = TextTrimmer()
        self.emoji_slang_handler = EmojiSlangHandler(text_columns)
        self.tokenizer = Tokenizer()
        self.lemmatizer = Lemmatizer(text_columns)
        self.stemmer = Stemmer()
        self.stopword_remover = StopwordRemover()
        self.normalizer = Normalizer()

    def remove_duplicates(self):
        """Removes duplicate entries."""
        self.data = self.duplicate_remover.process(self.data)

    def handle_missing_values(self):
        """Handles missing values in critical columns."""
        self.data = self.missing_value_handler.process(self.data)

    def normalize_text(self):
        """Normalizes text (lowercasing, trimming, etc.)."""
        self.data = self.normalizer.process(self.data, self.text_columns)
        self.data = self.text_trimmer.process(self.data, self.text_columns)

    def handle_slang_and_emojis(self):
        """Handles slangs and emojis in text columns."""
        self.data = self.emoji_slang_handler.process(self.data)

    def lemmatize(self):
        """Applies lemmatization."""
        self.data = self.lemmatizer.process(self.data)

    def remove_stopwords(self):
        """Removes stopwords."""
        self.data = self.stopword_remover.process(self.data, self.text_columns)

    def stem_words(self):
        """Applies stemming."""
        self.data = self.stemmer.process(self.data, self.text_columns)
