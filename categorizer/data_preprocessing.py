import pandas as pd
import re
import os
from datetime import datetime

def preprocess_data(df, output_folder=None):
    """
    Comprehensive data preprocessing, including:
    1. Remove special characters and non-ASCII characters
    2. Remove deleted/removed posts
    3. Remove duplicate posts
    4. Standardize dates
    5. Clean text fields
    
    Parameters:
    - df: Input DataFrame
    - output_folder: Optional, folder for saving intermediate results
    
    Returns:
    - Preprocessed DataFrame
    """
    print("Starting data preprocessing...")
    original_count = len(df)
    
    # Create output directory (if provided)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        df.to_csv(os.path.join(output_folder, "0_original_data.csv"), index=False)
    
    # 1. Remove deleted/removed posts
    print("Removing deleted/removed posts...")
    deleted_patterns = ['[deleted by user]', '[deleted]', '[removed]']
    if 'title' in df.columns:
        mask_deleted = ~df['title'].isin(deleted_patterns)
        df = df[mask_deleted].copy()
    
    deleted_count = original_count - len(df)
    print(f"Removed {deleted_count} deleted posts")
    
    # 2. Remove duplicate posts
    print("Removing duplicate posts...")
    if 'title' in df.columns:
        pre_dedup_count = len(df)
        df = df.drop_duplicates(subset=['title']).copy()
        dedup_count = pre_dedup_count - len(df)
        print(f"Removed {dedup_count} duplicate posts")
    
    # 3. Clean special characters in text
    print("Cleaning text fields...")
    text_columns = ['title', 'selftext']
    for col in text_columns:
        if col in df.columns:
            # Remove non-ASCII characters
            df[col] = df[col].apply(lambda x: clean_text(x) if pd.notna(x) else x)
    
    # 4. Standardize dates
    if 'date' in df.columns:
        print("Standardizing dates...")
        df['date'] = df['date'].apply(lambda x: standardize_date(x) if pd.notna(x) else x)
    
    # 5. Combine title and selftext
    if 'title' in df.columns and 'selftext' in df.columns:
        print("Combining title and body text...")
        df['combined_text'] = df.apply(
            lambda row: f"{row['title']} {row['selftext']}" if pd.notna(row['selftext']) else row['title'],
            axis=1
        )
    
    # Save preprocessed data
    if output_folder:
        df.to_csv(os.path.join(output_folder, "1_preprocessed_data.csv"), index=False)
    
    # Generate preprocessing statistics
    stats = {
        'Original row count': original_count,
        'Deleted posts': deleted_count,
        'Duplicate posts': dedup_count if 'title' in df.columns else 'N/A',
        'Rows after preprocessing': len(df),
        'Total cleanup rate': f"{((original_count - len(df)) / original_count * 100):.2f}%"
    }
    
    print("\nPreprocessing statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    if output_folder:
        with open(os.path.join(output_folder, "preprocessing_stats.txt"), 'w') as f:
            f.write("Preprocessing Statistics:\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    
    return df

def clean_text(text):
    """Clean special characters from text"""
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Replace newlines and tabs
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Filter out non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    return text.strip()

def standardize_date(date_str):
    """Standardize date format"""
    try:
        if str(date_str).isdigit():
            return datetime.fromtimestamp(int(date_str)).strftime('%Y-%m-%d %H:%M:%S')
        return pd.to_datetime(date_str).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return date_str