import pandas as pd

def analyze_useful_posts(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Total number of posts
    total_posts = len(df)
    
    # Filter posts that are useful to the government
    useful_posts = df[df['Intent Category'] == 'Yes']
    num_useful_posts = len(useful_posts)
    
    # Count the occurrences of each domain category for useful posts
    domain_counts = useful_posts['Domain Category'].value_counts()
    
    # Total number of unique domain categories
    total_unique_domains = domain_counts.shape[0]
    
    # Print the results
    print(f"Total number of posts: {total_posts}")
    print(f"Number of useful posts for the government: {num_useful_posts}")
    print(f"Total number of unique domain categories: {total_unique_domains}")
    print("\nTop 30 Domain Category Counts for Useful Posts:")
    print(domain_counts.head(30))

# Example usage
analyze_useful_posts('data/2023_post_processed.csv') 