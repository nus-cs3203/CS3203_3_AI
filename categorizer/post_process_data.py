import pandas as pd

def post_process_data(input_csv=None, output_csv=None, df=None):
    if input_csv is not None:
        # Read the CSV file
        df = pd.read_csv(input_csv)
    elif df is None:
        raise ValueError("Either input_csv or df must be provided")
    
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
    
    if output_csv:
        # Save the processed DataFrame to a new CSV file
        df.to_csv(output_csv, index=False)
        print(f"Post-processing complete. Results saved to {output_csv}")
    
    return df

# Example usage
# post_process_data(input_csv='data/2023_categorized_chunked2.csv', output_csv='data/2023_post_processed.csv')
# df = pd.read_csv('data/2023_categorized_chunked2.csv')
# processed_df = post_process_data(df=df)

# Example usage
#post_process_data('data/2023_categorized_chunked2.csv', 'data/2023_post_processed.csv') 