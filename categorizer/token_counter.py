import tiktoken
import pandas as pd
from tqdm import tqdm

def count_tokens(text):
    """Count tokens in a text string using GPT tokenizer"""
    encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoder.encode(str(text)))

def analyze_csv(file_path, sample_size=None):
    """Analyze token counts in a CSV file"""
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    total_rows = len(df)
    
    if sample_size:
        print(f"Sampling {sample_size} rows from {total_rows} total rows")
        df = df.sample(n=min(sample_size, len(df)))
    
    # Combine title and selftext
    df['combined_text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
    
    # Count tokens
    print("Counting tokens...")
    token_counts = []
    for text in tqdm(df['combined_text']):
        tokens = count_tokens(text)
        token_counts.append(tokens)
    
    # Calculate statistics
    avg_tokens = sum(token_counts) / len(token_counts)
    max_tokens = max(token_counts)
    total_tokens = sum(token_counts)
    
    if sample_size:
        # Estimate total based on sample
        total_tokens = (total_tokens / len(df)) * total_rows
    
    # Print report
    print(f"\n=== Token Count Report for {file_path} ===")
    print(f"Total rows: {total_rows:,}")
    print(f"Average tokens per row: {int(avg_tokens):,}")
    print(f"Max tokens in any row: {max_tokens:,}")
    print(f"Estimated total tokens: {int(total_tokens):,}")

if __name__ == "__main__":
    # Analyze files
    files = [
        "200rows_Filtered_Data.csv",
        "2023_filtered_data.csv"
    ]
    
    for file_path in files:
        try:
            # Use sampling for large files
            sample_size = 100 if "2023" in file_path else None
            analyze_csv(file_path, sample_size=sample_size)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}") 