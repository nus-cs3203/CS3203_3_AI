import pandas as pd
import os
from news_filter import filter_for_opinions
from r1_categorizer import categorize_complaints

def process_posts(input_file, output_folder, num_rows=1000):
    """
    Process posts through both news filter and complaint categorizer
    
    Args:
        input_file: Path to input CSV file
        output_folder: Path to output folder
        num_rows: Number of rows to process (default 1000)
    """
    # Create output directory and subdirectories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "news_filter_results"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "complaint_categorizer_results"), exist_ok=True)
    
    print("Step 1: Reading and filtering news posts...")
    # First apply news filter
    df = pd.read_csv(input_file, nrows=num_rows)
    
    # Save initial dataset
    df.to_csv(os.path.join(output_folder, "1_initial_posts.csv"), index=False)
    
    # Apply news filter
    df_filtered = filter_for_opinions(
        df=df,
        output_folder=os.path.join(output_folder, "news_filter_results")
    )
    
    # Save posts after news filtering
    df_filtered.to_csv(os.path.join(output_folder, "2_after_news_filter.csv"), index=False)
    
    print("\nStep 2: Categorizing complaints...")
    # Then apply complaint categorizer
    df_categorized = categorize_complaints(
        df=df_filtered,
        output_csv=os.path.join(output_folder, "complaint_categorizer_results/categorized_posts.csv")
    )
    
    # Save final results with clear naming
    df_categorized.to_csv(os.path.join(output_folder, "3_final_categorized_posts.csv"), index=False)
    
    # Save only the relevant complaints
    df_relevant_complaints = df_categorized[df_categorized['Intent Category'].str.lower() == 'yes']
    df_relevant_complaints.to_csv(os.path.join(output_folder, "4_relevant_complaints_only.csv"), index=False)
    
    # Generate summary statistics
    stats = {
        'Initial posts': len(df),
        'Posts after news filter': len(df_filtered),
        'Posts with relevant complaints': len(df_relevant_complaints),
        'Domain categories distribution': df_relevant_complaints['Domain Category'].value_counts().to_dict(),
        'Average sentiment score': df_categorized['Sentiment Score'].mean(),
        'Average importance level': df_categorized['Importance Level'].mean()
    }
    
    # Save statistics
    with open(os.path.join(output_folder, "processing_summary.txt"), 'w') as f:
        f.write("Processing Statistics:\n\n")
        for key, value in stats.items():
            f.write(f"{key}:\n{value}\n\n")
    
    print("\nProcessing complete!")
    print(f"Results saved to: {output_folder}")
    print("\nFiles generated:")
    print("1. 1_initial_posts.csv - Initial 1000 posts")
    print("2. 2_after_news_filter.csv - Posts after filtering out news")
    print("3. 3_final_categorized_posts.csv - All posts with categorization")
    print("4. 4_relevant_complaints_only.csv - Only the relevant complaints")
    print("5. processing_summary.txt - Summary statistics")
    print("\nSubfolders:")
    print("- news_filter_results/ - Detailed results from news filtering")
    print("- complaint_categorizer_results/ - Detailed results from categorization")
    
    return df_categorized

if __name__ == "__main__":
    input_file = "last_round_files/all_posts_2022_2025.csv"
    output_folder = "last_round_files/processed_posts_5000"
    
    df_final = process_posts(
        input_file=input_file,
        output_folder=output_folder,
        num_rows=5000
    ) 