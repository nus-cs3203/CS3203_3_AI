import pandas as pd
from datetime import datetime
import os

def standardize_date(date_str):
    try:
        if str(date_str).isdigit():
            return datetime.fromtimestamp(int(date_str)).strftime('%Y-%m-%d %H:%M:%S')
        return pd.to_datetime(date_str).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return None

def post_process_data(input_csv=None, output_csv=None, df=None):
    if input_csv is not None:
        if not os.path.exists(input_csv):
            raise FileNotFoundError("The input file does not exist")
        if os.path.getsize(input_csv) == 0:
            raise ValueError("The input file is empty")
        df = pd.read_csv(input_csv)
    elif df is None:
        raise ValueError("Either input_csv or df must be provided")
    
    # Standardize date format if a date column exists
    if 'date' in df.columns:
        df['date'] = df['date'].apply(standardize_date)
    
    # Define the list of valid domain categories
    valid_categories = [
        "Housing", "Healthcare", "Transportation", "Public Safety", 
        "Transport", "Education", "Environment", "Employment", 
        "Public Health", "Legal", "Economy", "Politics", "Technology", 
        "Infrastructure"
    ]
    
    # Replace domain categories not in the valid list with "Others"
    df['Domain Category'] = df['Domain Category'].apply(
        lambda x: x if x in valid_categories else "Others"
    )
    
    # Filter out entries where intent_category is 'No'
    if 'Intent Category' in df.columns:
        df = df[df['Intent Category'] == 'Yes']
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Post-processing complete. Results saved to {output_csv}")
    
    return df

# Example usage
# post_process_data(input_csv='data/2023_categorized_chunked2.csv', output_csv='data/2023_post_processed.csv')
# df = pd.read_csv('data/2023_categorized_chunked2.csv')
# processed_df = post_process_data(df=df)

# Example usage
#post_process_data('data/2023_categorized_chunked2.csv', 'data/2023_post_processed.csv') 