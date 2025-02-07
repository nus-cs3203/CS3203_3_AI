from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm
import html
import re
from datetime import datetime
import csv

# Configure device
device = 0  # Using GPU 0

try:
    # Use the specified classifier
    # model_name = "cross-encoder/nli-deberta-v3-large"
    # complaint_classifier = pipeline("zero-shot-classification",
    #                              model=model_name,
    #                              device=device)
    
    # domain_classifier = pipeline("zero-shot-classification",
    #                            model=model_name,
    #                            device=device)
    
    model_name = "facebook/bart-large-mnli"
    complaint_classifier = pipeline("zero-shot-classification", model=model_name)
    domain_classifier = pipeline("zero-shot-classification", model=model_name)

    
except Exception as e:
    print(f"Warning: Could not load DeBERTa model, falling back to BART: {str(e)}")
    model_name = "facebook/bart-large-mnli"
    complaint_classifier = pipeline("zero-shot-classification", model=model_name)
    domain_classifier = pipeline("zero-shot-classification", model=model_name)

# First level: More specific categories
complaint_categories = [
    "Direct Complaint",          # Explicit complaints/criticisms
    "General negative feelings/observations, but not complaint",        # General negative feelings/observations
    "Positive Feedback",         # Direct praise/appreciation
    "General positive feelings/observations, but not positive feedback",        # General positive feelings/observations
    "Neutral Discussion"         # Questions/discussions without strong sentiment
]

# Second level: Domain categories for complaints
domain_categories = [
    "Healthcare",           # Medical services, hospitals, health insurance
    "Education",           # Schools, universities, teaching quality
    "Transportation",      # Public transit, traffic, road conditions
    "Employment",          # Jobs, workplace issues, salary
    "Environmental",       # Pollution, cleanliness, sustainability
    "Public Safety",       # Crime, security, emergency services
    "Social Services",     # Government services, welfare, social support
    "Recreation",          # Parks, entertainment, sports facilities
    "Housing",            # Property, rental, maintenance
    "Food Services",      # Restaurants, food quality, delivery
    "Infrastructure",     # Roads, utilities, public facilities
    "Retail",            # Shopping, customer service, products
    "Technology",        # Internet, devices, digital services
    "Financial",         # Banking, costs, fees
    "Noise",             # Noise pollution, disturbances
]

def classify_post(text):
    """
    Two-level classification with better sentiment detection
    """
    # First level: Check category scores
    complaint_result = complaint_classifier(
        text,
        candidate_labels=complaint_categories,
        hypothesis_template="This post expresses {}", # Changed template to better capture sentiment
        multi_label=False
    )
    
    # Get scores for all categories
    category_scores = {
        label: score for label, score in 
        zip(complaint_result['labels'], complaint_result['scores'])
    }
    
    # Consider both direct complaints and strong negative sentiment
    is_complaint = complaint_result['labels'][0] in ["Direct Complaint", "Negative Sentiment"]
    complaint_confidence = complaint_result['scores'][0]
    
    # If it's a complaint with high confidence
    if is_complaint and complaint_confidence > 0.6:
        # Domain classification with multi-label
        domain_result = domain_classifier(
            text,
            candidate_labels=domain_categories,
            hypothesis_template="This is a complaint about {} related issues",
            multi_label=True
        )
        
        # Only consider domains with higher confidence (0.4)
        relevant_domains = [(label, score) for label, score in 
                          zip(domain_result['labels'], domain_result['scores']) 
                          if score > 0.4]
        
        if relevant_domains:
            relevant_domains.sort(key=lambda x: x[1], reverse=True)
            return {
                "is_complaint": True,
                "category_scores": category_scores,
                "complaint_confidence": complaint_confidence,
                "primary_domain": relevant_domains[0][0],
                "primary_domain_confidence": relevant_domains[0][1],
                "related_domains": [domain for domain, score in relevant_domains[1:]]
            }
    
    return {
        "is_complaint": False,
        "category_scores": category_scores,
        "confidence": complaint_confidence,
        "text": text
    }

def clean_text(text):
    """Clean the text by removing special characters and decoding HTML entities"""
    if pd.isna(text) or text == '[deleted]' or text == '[removed]':
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def process_reddit_post(title, selftext):
    """Process both title and selftext of a Reddit post"""
    # Clean and combine title and selftext
    clean_title = clean_text(title)
    clean_selftext = clean_text(selftext)
    
    # If both are empty after cleaning, skip classification
    if not clean_title and not clean_selftext:
        return None
        
    combined_text = f"{clean_title} {clean_selftext}".strip()
    if not combined_text:
        return None
        
    result = classify_post(combined_text)
    
    # Print detailed scores
    print("\nText:", combined_text[:100], "...")
    print("\nCategory Scores:")
    for category, score in result["category_scores"].items():
        print(f"{category}: {score:.3f}")
    
    if result["is_complaint"]:
        print(f"\nClassified as COMPLAINT with {result['complaint_confidence']:.3f} confidence")
        print(f"Primary domain: {result['primary_domain']} ({result['primary_domain_confidence']:.3f})")
        if result.get('related_domains'):
            print(f"Related domains: {', '.join(result['related_domains'])}")
    else:
        print("\nNot classified as complaint")
    
    result['original_title'] = title
    result['original_selftext'] = selftext
    return result

def process_csv_batch(input_file, output_file, batch_size=50):
    """Process the CSV file in batches"""
    print(f"Reading from {input_file}")
    
    # Read CSV in chunks
    df_chunks = pd.read_csv(input_file, chunksize=batch_size)
    
    # Create output file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{output_file}_{timestamp}.csv"
    
    # Modify headers to include analysis
    headers = ['original_title', 'original_selftext', 'is_complaint', 
              'complaint_confidence', 'primary_domain', 
              'primary_domain_confidence', 'related_domains']
    
    # Create a list to store all results for analysis
    all_results = []
    domain_counts = {}
    total_processed = 0
    total_complaints = 0
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        # Process each chunk
        for chunk in tqdm(df_chunks, desc="Processing posts"):
            for _, row in chunk.iterrows():
                try:
                    result = process_reddit_post(
                        str(row['title']) if pd.notna(row['title']) else "",
                        str(row['selftext']) if pd.notna(row['selftext']) else ""
                    )
                    
                    if result is None:
                        continue
                        
                    # Prepare row for writing
                    output_row = {
                        'original_title': row['title'],
                        'original_selftext': row['selftext'],
                        'is_complaint': result['is_complaint'],
                        'complaint_confidence': result.get('complaint_confidence', None),
                        'primary_domain': result.get('primary_domain', None),
                        'primary_domain_confidence': result.get('primary_domain_confidence', None),
                        'related_domains': ','.join(result.get('related_domains', []))
                    }
                    
                    writer.writerow(output_row)
                    total_processed += 1
                    
                    if result['is_complaint']:
                        total_complaints += 1
                        # Track domain statistics
                        if 'primary_domain' in result:
                            domain = result['primary_domain']
                            domain_counts[domain] = domain_counts.get(domain, 0) + 1
                        if 'related_domains' in result:
                            for domain in result.get('related_domains', []):
                                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    
                    # Print progress every 50 posts
                    if total_processed % 50 == 0:
                        print(f"\nProcessed {total_processed} posts, "
                              f"Found {total_complaints} complaints")
                    
                except Exception as e:
                    print(f"\nError processing row: {e}")
                    continue
        
        # After processing all posts, add analysis summary rows
        writer.writerow({'original_title': '=== ANALYSIS SUMMARY ==='})
        writer.writerow({
            'original_title': 'Total Posts',
            'original_selftext': str(total_processed)
        })
        writer.writerow({
            'original_title': 'Total Complaints',
            'original_selftext': str(total_complaints),
            'complaint_confidence': f"{(total_complaints/total_processed)*100:.2f}%"
        })
        
        writer.writerow({'original_title': '=== DOMAIN DISTRIBUTION ==='})
        for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
            writer.writerow({
                'original_title': domain,
                'original_selftext': str(count),
                'complaint_confidence': f"{(count/total_complaints)*100:.2f}% of complaints"
            })

if __name__ == "__main__":
    input_file = "200rows.csv"  # Changed to your test file
    output_file = "bart_output200.csv"
    
    print("Starting classification process...")
    process_csv_batch(input_file, output_file)
    print("Classification complete!")