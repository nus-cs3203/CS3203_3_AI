import pandas as pd

from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.polarity.advanced import AdvancedSentimentClassifier

# Load dataset
df = pd.read_csv("tests/sentiment_analyser/data/2022_2025_merged_llm_pre_500.txt")
classifier =  AdvancedSentimentClassifier()
context = SentimentAnalysisContext(classifier)
df_result = context.analyze(df, text_cols=["title_with_desc"])
df_result = df_result[["title_with_desc_label", "title_with_desc_score"]]  
    
df_result.to_csv("tests/sentiment_analyser/data/2022_2025_merged_llm_pre_500_advanced.txt", index=False)