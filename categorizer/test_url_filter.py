import pandas as pd
import time
from news_filter import is_news_url

def test_url_filter(input_csv, output_folder):
    """Test URL-based filtering on first 1000 rows"""
    start_time = time.time()
    
    # Read first 1000 rows of the CSV file
    df = pd.read_csv(input_csv, nrows=1000)
    
    initial_count = len(df)
    print(f"Initial number of rows: {initial_count}")
    
    # First filter: Remove posts from news websites if URL is available
    if 'url' in df.columns:
        df['is_news_url'] = df['url'].apply(is_news_url)
        news_by_url = df['is_news_url'].sum()
        print(f"Found {news_by_url} posts from news websites")
        
        # Save news URLs for inspection
        df[df['is_news_url']].to_csv(f"{output_folder}/news_urls.csv", index=False)
        
        # Save non-news URLs
        df_non_news = df[~df['is_news_url']].copy()
        df_non_news.to_csv(f"{output_folder}/non_news_urls.csv", index=False)
        
        url_filtered_count = len(df_non_news)
        print(f"Remaining posts after URL filtering: {url_filtered_count}")
        
        # Save statistics
        stats = {
            'Initial posts': initial_count,
            'News posts': news_by_url,
            'Non-news posts': url_filtered_count,
            'News percentage': f"{(news_by_url/initial_count)*100:.2f}%"
        }
        
        with open(f"{output_folder}/filter_stats.txt", 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    else:
        print("No URL column found in the CSV file")
    
    end_time = time.time()
    print(f"Total time taken: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    input_file = "last_round_files/all_posts_2022_2025.csv"  
    output_folder = "last_round_files/facts_filter" 
    
    # Create output folder if it doesn't exist
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    test_url_filter(input_file, output_folder) 