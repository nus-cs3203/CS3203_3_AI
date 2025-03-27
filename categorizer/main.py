import pandas as pd
import os
from news_filter import filter_for_opinions
from r1_categorizer import categorize_complaints

def process_posts(input_file, output_folder, num_rows=200):
    """
    Process posts through news filter and two rounds of complaint categorization
    """
    # Create directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "news_filter_results"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "complaint_categorizer_results"), exist_ok=True)
    
    print("Step 1: Reading and filtering news posts...")
    # Get total number of rows
    total_rows = sum(1 for _ in open(input_file)) - 1  # -1 for header
    
    # Calculate how many rows to skip
    skip_rows = max(0, total_rows - num_rows)
    
    # Read the last num_rows rows
    df = pd.read_csv(input_file, skiprows=range(1, skip_rows + 1) if skip_rows > 0 else None)
    
    print(f"Processing the last {num_rows} rows from total {total_rows} rows")
    df.to_csv(os.path.join(output_folder, "1_initial_posts.csv"), index=False)
    
    # First filter: news filter
    df_filtered = filter_for_opinions(
        df=df,
        output_folder=os.path.join(output_folder, "news_filter_results")
    )
    df_filtered.to_csv(os.path.join(output_folder, "2_after_news_filter.csv"), index=False)
    
    print("\nStep 2: First round complaint categorization...")
    # First round categorization with default batch size (50)
    df_categorized = categorize_complaints(
        df=df_filtered,
        output_csv=os.path.join(output_folder, "complaint_categorizer_results/first_round_categorized.csv"),
        is_second_round=False
    )
    
    # Filter out non-complaints after first round
    df_first_round_complaints = df_categorized[
        df_categorized['Intent Category'].str.lower() == 'yes'
    ].copy()
    
    print(f"\nFound {len(df_first_round_complaints)} complaints in first round")
    df_first_round_complaints.to_csv(os.path.join(output_folder, "3_first_round_complaints.csv"), index=False)
    
    # Prepare context for second round verification
    df_first_round_complaints['combined_text'] = df_first_round_complaints.apply(
        lambda row: f"""Original post: {row['combined_text']}
First round analysis:
- Category: {row['Domain Category']}
- Confidence: {row['Confidence Score']}
- Sentiment: {row['Sentiment Score']}
- Importance: {row['Importance Level']}

Please verify if this is truly a government-relevant complaint that requires policy action.""",
        axis=1
    )
    
    print("\nStep 3: Second round verification of first round complaints...")
    # Second round verification with smaller batch size (10) for more accurate analysis
    df_verified = categorize_complaints(
        df=df_first_round_complaints,
        output_csv=os.path.join(output_folder, "complaint_categorizer_results/second_round_verified.csv"),
        is_second_round=True  # This triggers the smaller batch size
    )
    
    # Final filtering: keep only those verified as complaints in second round
    df_final_complaints = df_verified[
        df_verified['Intent Category'].str.lower() == 'yes'
    ].copy()
    
    df_final_complaints.to_csv(os.path.join(output_folder, "4_final_verified_complaints.csv"), index=False)
    
    # Generate summary statistics
    stats = {
        'Initial posts': len(df),
        'Posts after news filter': len(df_filtered),
        'First round total processed': len(df_categorized),
        'First round complaints': len(df_first_round_complaints),
        'Final verified complaints': len(df_final_complaints),
        'Domain categories distribution': df_final_complaints['Domain Category'].value_counts().to_dict(),
        'Average confidence (final)': df_final_complaints['Confidence Score'].mean(),
        'Average sentiment (final)': df_final_complaints['Sentiment Score'].mean(),
        'Average importance (final)': df_final_complaints['Importance Level'].mean()
    }
    
    # Save statistics
    with open(os.path.join(output_folder, "processing_summary.txt"), 'w') as f:
        f.write("Processing Statistics:\n\n")
        for key, value in stats.items():
            f.write(f"{key}:\n{value}\n\n")
    
    print("\nProcessing complete!")
    print(f"Results saved to: {output_folder}")
    print("\nFiles generated:")
    print("1. 1_initial_posts.csv - Initial posts")
    print("2. 2_after_news_filter.csv - Posts after filtering out news")
    print("3. complaint_categorizer_results/first_round_categorized.csv - First round results")
    print("4. complaint_categorizer_results/second_round_verified.csv - Second round results")
    print("5. 4_final_verified_complaints.csv - Posts classified as complaints in both rounds")
    print("6. processing_summary.txt - Summary statistics")
    
    return df_final_complaints

if __name__ == "__main__":
    input_file = "last_round_files/all_posts_2022_2025.csv"
    output_folder = "last_round_files/processed_posts_3000_last"
    
    # Get total number of rows in the file
    total_rows = sum(1 for _ in open(input_file)) - 1  # -1 for header
    
    # Calculate how many rows to skip to get the last 3000 rows
    skip_rows = max(0, total_rows - 3000)
    
    # Read the last 3000 rows
    df_final = process_posts(
        input_file=input_file,
        output_folder=output_folder,
        num_rows=3000
    ) 