import pandas as pd

def analyze_categorized_data(input_csv, output_txt):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Count the occurrences of each intent category
    intent_counts = df['Intent Category'].value_counts()
    
    # Count the occurrences of each domain category within "Direct Complaint"
    direct_complaint_df = df[df['Intent Category'] == 'Direct Complaint']
    domain_counts_within_complaints = direct_complaint_df['Domain Category'].value_counts()
    
    # Write the analysis to a text file
    with open(output_txt, 'w') as f:
        f.write("Intent Category Counts:\n")
        f.write(intent_counts.to_string())
        f.write("\n\nDomain Category Counts within 'Direct Complaint':\n")
        f.write(domain_counts_within_complaints.to_string())
    
    print("Analysis complete. Results saved to", output_txt)

# Example usage
analyze_categorized_data('data/500rows_categorized.csv', 'data/analysis_results.txt') 