import pandas as pd

from categorizer.deepseek_categorizer_chunked import categorize_complaints
from categorizer.post_process_data import post_process_data

def categorize_and_process(df: pd.DataFrame) -> pd.DataFrame:
    input_csv = "temp_input.csv"
    output_categorized_csv = "temp_categorized.csv"
    output_final_csv = "temp_final.csv"

    df.to_csv(input_csv, index=False)
    
    # Categorize complaints
    categorize_complaints(input_csv, output_categorized_csv)
    
    # Post-process data
    post_process_data(output_categorized_csv, output_final_csv)
    
    # Load final processed data
    final_df = pd.read_csv(output_final_csv)
    
    return final_df

# Example usage
df = pd.read_csv("files/sentiment_scored_2023_data.csv")
result_df = categorize_and_process(df)
result_df.to_csv("files/categorized_and_processed_data.csv", index=False)
