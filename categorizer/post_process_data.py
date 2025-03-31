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
    if 'Domain Category' in df.columns:
        df['Domain Category'] = df['Domain Category'].apply(clean_category)
    
    # Define the list of valid domain categories
    valid_categories = [
        "Housing", "Healthcare", "Public Safety", "Transport",
        "Education", "Environment", "Employment", "Public Health",
        "Legal", "Economy", "Politics", "Technology",
        "Infrastructure", "Others"
    ]
    
    # Replace domain categories not in the valid list with "Others"
    if 'Domain Category' in df.columns:
        df['Domain Category'] = df['Domain Category'].apply(
            lambda x: x if x in valid_categories else "Others"
        )
    
    # Filter out entries where intent_category is 'No'
    if 'Intent Category' in df.columns:
        df = df[df['Intent Category'].str.lower() == 'yes']
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    # Remove duplicate complaints with the same title
    if 'title' in df.columns:
        df = df.drop_duplicates(subset='title', keep='first')

    # Handle id field: use name if available, otherwise use id
    if 'name' in df.columns:
        df['id'] = df['name']
    # else keep existing id if it exists

    # Handle description field: use selftext if available
    if 'selftext' in df.columns:
        df['description'] = df['selftext']
    # else keep existing description if it exists

    # Handle category field: use Domain Category if available
    if 'Domain Category' in df.columns:
        df['category'] = df['Domain Category']
    # else keep existing category if it exists

    # Map other fields
    df['sentiment_by_vader'] = df['combined_text_score']
    df['sentiment'] = df.get('Sentiment Score', 0.0)
    df['importance'] = df.get('Importance Level', 0.0)
    df['source'] = "Reddit"
    df['confidence'] = df.get('Confidence Score', 0.0)
    
    # Select and reorder columns to match the schema
    output_columns = ['id', 'title', 'description', 'category', 
                     'sentiment', 'confidence', 'importance', 
                     'sentiment_by_vader', 'url', 'source', 'date']
    
    # Ensure all required columns exist
    for col in output_columns:
        if col not in df.columns:
            df[col] = '' if col not in ['sentiment', 'confidence', 'importance', 'sentiment_by_vader'] else 0.0
    
    output_df = df[output_columns]

    for col in df.select_dtypes(include=['float']).columns:
        df[col] = df[col].fillna(0.0) 
        df[col] = df[col].replace([float('inf'), float('-inf')], 0.0)  

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