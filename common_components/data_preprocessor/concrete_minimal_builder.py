from common_components.data_preprocessor.builder import PreprocessorBuilder
from common_components.data_preprocessor.components.columns_joiner import ColumnsJoiner
from common_components.data_preprocessor.components.duplicate_remover import DuplicateRemover
from common_components.data_preprocessor.components.missing_values_handler import MissingValueHandler
from common_components.data_preprocessor.components.normalizer import Normalizer
from common_components.data_preprocessor.components.stopword_remover import StopwordRemover
from common_components.data_preprocessor.components.text_trimmer import TextTrimmer

class MinimalPreprocessorBuilder(PreprocessorBuilder):
    """Concrete Builder implementing preprocessing steps for minimal preprocessing."""

    def __init__(self, critical_columns, data, subset = None):
        self.critical_columns = critical_columns
        self.subset = subset if subset else critical_columns
        self.data = data
        self.reset()

    def reset(self):
        """Resets the preprocessing components."""
        self.columns_joiner = ColumnsJoiner(subset=self.subset)
        self.duplicate_remover = DuplicateRemover(subset=self.critical_columns)
        self.missing_value_handler = MissingValueHandler(critical_columns=self.critical_columns)

    def remove_duplicates(self):
        """Removes duplicate entries."""
        self.data = self.duplicate_remover.process(self.data)

    def handle_missing_values(self):
        """Handles missing values in critical columns."""
        self.data = self.missing_value_handler.process(self.data)

    def perform_preprocessing(self):
        """Executes the preprocessing steps."""
        self.remove_duplicates()
        self.handle_missing_values()
        self.columns_joiner.process(self.data)
        
    def get_result(self):
        """Returns the preprocessed data."""
        return self.data