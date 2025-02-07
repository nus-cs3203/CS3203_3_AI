import pandas as pd

def analyze_complaints(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Total number of posts
    total_posts = len(df)
    
    # Number of complaints
    complaints = df[df['is_complaint'] == True]
    num_complaints = len(complaints)
    
    # Domain analysis
    domain_counts = {}
    for _, row in complaints.iterrows():
        if pd.notna(row['primary_domain']):
            domain = row['primary_domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
        # Count related domains if they exist
        if pd.notna(row['related_domains']) and row['related_domains']:
            related = row['related_domains'].split(',')
            for domain in related:
                domain = domain.strip()
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"\nAnalysis Results for {file_path}")
    print("-" * 50)
    print(f"Total posts analyzed: {total_posts}")
    print(f"Total complaints identified: {num_complaints} ({(num_complaints/total_posts)*100:.2f}%)")
    print("\nDomain Distribution:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{domain}: {count} posts ({(count/num_complaints)*100:.2f}% of complaints)")

# Analyze the most recent output file
analyze_complaints("output500.csv_20250207_122446.csv") 