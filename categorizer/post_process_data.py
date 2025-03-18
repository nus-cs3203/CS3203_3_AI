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

    # Function to clean category names by removing part after a slash
    def clean_category(category):
        primary_category = category.split('/')[0].strip()
        return primary_category.title()

    # Apply the cleaning function to 'Domain Category'
    df['Domain Category'] = df['Domain Category'].apply(clean_category)
    
    # Define the list of valid domain categories
    valid_categories = [
        "Housing", "Healthcare", "Public Safety", "Transport",
        "Education", "Environment", "Employment", "Public Health",
        "Legal", "Economy", "Politics", "Technology",
        "Infrastructure", "Others"
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
    
    # Ensure the output matches the schema
    df['id'] = df['id']  # Use 'id' from the API response
    df['category'] = df['Domain Category']
    df['date'] = df['date']
    df['sentiment'] = df['title_with_desc_score']
    df['source'] = "Reddit"
    df['description'] = df['selftext']
    df['confidence'] = df['Confidence Score']  # Include confidence score
    
    # Select and reorder columns to match the schema
    output_df = df[['id', 'title', 'description', 'category', 'date', 'sentiment', 'url', 'source', 'confidence']]

    if output_csv:
        output_df.to_csv(output_csv, index=False)
        print(f"Post-processing complete. Results saved to {output_csv}")
    
    return output_df

# Example usage
# post_process_data(input_csv='data/2023_categorized_chunked2.csv', output_csv='data/2023_post_processed.csv')
# df = pd.read_csv('data/2023_categorized_chunked2.csv')
# processed_df = post_process_data(df=df)

# Example usage
#post_process_data('data/2023_categorized_chunked2.csv', 'data/2023_post_processed.csv') 