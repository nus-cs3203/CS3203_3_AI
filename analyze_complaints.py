import pandas as pd
import argparse
from pathlib import Path

def analyze_complaints(csv_file, output_format='text', save_to_file=None):
    """
    Analyze complaints from CSV file
    
    Args:
        csv_file (str): Path to CSV file
        output_format (str): 'text' or 'json' or 'csv'
        save_to_file (str): Optional file path to save results
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Add row numbers from original file
    df['original_row'] = df.index + 1
    
    # Filter only complaints
    complaints_df = df[df['is_complaint'] == True].copy()
    
    # Convert confidence columns to float
    complaints_df['complaint_confidence'] = pd.to_numeric(complaints_df['complaint_confidence'], errors='coerce')
    complaints_df['primary_domain_confidence'] = pd.to_numeric(complaints_df['primary_domain_confidence'], errors='coerce')
    
    
    # Create a more readable format
    complaints_summary = complaints_df.apply(lambda row: {
        'title': row['original_title'],
        'text': row['original_selftext'] if pd.notna(row['original_selftext']) else '',
        'confidence': f"{row['complaint_confidence']:.2%}" if pd.notna(row['complaint_confidence']) else 'N/A',
        'primary_domain': row['primary_domain'],
        'domain_confidence': f"{row['primary_domain_confidence']:.2%}" if pd.notna(row['primary_domain_confidence']) else 'N/A',
        'related_domains': row['related_domains'] if pd.notna(row['related_domains']) else ''
    }, axis=1).tolist()
    
    # Print complaints in a readable format
    print(f"Total posts analyzed: {len(df)}")
    print(f"Total complaints found: {len(complaints_df)} ({len(complaints_df)/len(df):.2%})")
    print("\n=== COMPLAINTS SUMMARY ===")
    
    for i, complaint in enumerate(complaints_summary, 1):
        print(f"\n{i}. Complaint in {complaint['primary_domain']} ({complaint['domain_confidence']} confidence)")
        print(f"Title: {complaint['title']}")
        if complaint['text']:
            print(f"Text: {complaint['text'][:200]}...")
        print(f"Complaint confidence: {complaint['confidence']}")
        if complaint['related_domains']:
            print(f"Related domains: {complaint['related_domains']}")
        print("-" * 80)
    
    # Domain analysis
    domain_counts = complaints_df['primary_domain'].value_counts()
    print("\n=== DOMAIN DISTRIBUTION ===")
    for domain, count in domain_counts.items():
        print(f"{domain}: {count} complaints ({count/len(complaints_df):.2%})")
    
    # Save results
    if save_to_file:
        output_path = Path(save_to_file)
        base_path = output_path.parent / output_path.stem
        
        # Save full complaints data
        if output_format == 'csv':
            complaints_df.to_csv(f"{base_path}_full.csv", index=False)
            
            # Save simplified version with key information
            simplified_df = complaints_df[[
                'original_row',
                'original_title',
                'original_selftext',
                'complaint_confidence',
                'primary_domain',
                'primary_domain_confidence',
                'related_domains'
            ]].copy()
            simplified_df.to_csv(f"{base_path}_simplified.csv", index=False)
            
        elif output_format == 'json':
            complaints_df.to_json(f"{base_path}_full.json", orient='records')
            
        print(f"\nResults saved to:")
        print(f"Full data: {base_path}_full.{output_format}")
        if output_format == 'csv':
            print(f"Simplified data: {base_path}_simplified.csv")

def main():
    parser = argparse.ArgumentParser(description='Analyze complaints from CSV file')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--format', choices=['text', 'json', 'csv'], 
                       default='csv', help='Output format')
    parser.add_argument('--save', help='Save results to file')
    
    args = parser.parse_args()
    analyze_complaints(args.csv_file, args.format, args.save)

if __name__ == "__main__":
    main() 