from abc import ABC, abstractmethod
import pandas as pd

class SentimentClassifier(ABC):
    """Strategy Interface - Defines a method for sentiment classification on Pandas DataFrames."""
    
    @abstractmethod
    def classify(self, df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
        """
        Applies sentiment analysis to specified columns in a Pandas DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            text_cols (list): List of column names to perform sentiment analysis on.

        Returns:
            pd.DataFrame: DataFrame with additional sentiment-related columns.
        """
        pass
