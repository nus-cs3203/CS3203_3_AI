from common_components.data_preprocessor.builder import PreprocessorBuilder
from common_components.data_preprocessor.components.columns_joiner import ColumnsJoiner
from common_components.data_preprocessor.components.duplicate_remover import DuplicateRemover
from common_components.data_preprocessor.components.missing_values_handler import MissingValueHandler
from common_components.data_preprocessor.components.normalizer import Normalizer
from common_components.data_preprocessor.components.stopword_remover import StopwordRemover
from common_components.data_preprocessor.components.text_trimmer import TextTrimmer

class GeneralPreprocessorBuilder(PreprocessorBuilder):
    """Concrete Builder implementing preprocessing steps for general preprocessing."""

    def __init__(self, critical_columns, text_columns, data, subset = None):
        self.critical_columns = critical_columns
        self.text_columns = text_columns
        self.subset = subset if subset else critical_columns
        self.data = data
        self.reset()

    def reset(self):
        """Resets the preprocessing components."""
        self.columns_joiner = ColumnsJoiner(subset=self.subset)
        self.duplicate_remover = DuplicateRemover(subset=self.critical_columns)
        self.missing_value_handler = MissingValueHandler(critical_columns=self.critical_columns)
        self.text_trimmer = TextTrimmer(text_columns=self.text_columns)
        self.normalizer = Normalizer(text_columns=self.text_columns)

    def join_columns(self):
        """Joins text columns."""
        self.data = self.columns_joiner.process(self.data)

    def remove_duplicates(self):
        """Removes duplicate entries."""
        self.data = self.duplicate_remover.process(self.data)

    def handle_missing_values(self):
        """Handles missing values in critical columns."""
        self.data = self.missing_value_handler.process(self.data)

    def normalize_text(self):
        """Normalizes text (lowercasing, trimming, etc.)."""
        self.data = self.normalizer.process(self.data)

    def trim_text(self):
        """Trims text columns."""
        self.data = self.text_trimmer.process(self.data)

    def perform_preprocessing(self):
        """Executes the preprocessing steps."""
        self.remove_duplicates()
        self.handle_missing_values()
        self.join_columns()
        self.normalize_text()
        self.trim_text()
        self.data.reset_index(drop=True, inplace=True)

    def get_result(self):
        """Returns the preprocessed data."""
        return self.data