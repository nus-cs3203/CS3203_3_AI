import pandas as pd

def post_process_data(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Define the list of valid domain categories
    valid_categories = [
        "Housing", "Healthcare", "Transportation", "Public Safety", 
        "Transport", "Education", "Environment", "Employment", 
        "Public Health", "Legal", "Economy", "Politics", "Technology", 
        "Infrastructure"
    ]
    
    # Replace domain categories not in the valid list with "Unknown"
    df['Domain Category'] = df['Domain Category'].apply(
        lambda x: x if x in valid_categories else "Others"
    )
    
    # Save the processed DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Post-processing complete. Results saved to {output_csv}")

# Example usage
#post_process_data('data/2023_categorized_chunked2.csv', 'data/2023_post_processed.csv') 