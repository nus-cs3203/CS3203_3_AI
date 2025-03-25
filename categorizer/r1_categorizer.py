from openai import OpenAI
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

client = OpenAI(
    api_key="af598933-afc1-4095-853b-c00879ec86d0",
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

def process_batch(batch_texts, categories=None):
    if categories is None:
        categories = [
            "Housing", "Healthcare", "Public Safety", "Transport",
            "Education", "Environment", "Employment", "Public Health",
            "Legal", "Economy", "Politics", "Technology",
            "Infrastructure", "Others"
        ]
    
    categories_str = ", ".join(categories)
    
    # Prepare the system message
    system_message = {
        "role": "system", 
        "content": "You are a helpful assistant categorizing complaints."
    }
    
    # Calculate the number of entries in the current batch
    num_entries = len(batch_texts)
    
    # Define the user instruction with dynamic entry count
    user_instruction = {
        "role": "user", 
        "content": f"""
            You will be given Reddit posts from r/Singapore. Your job is to classify whether the post describes a complaint that is relevant to government authorities.

            A post should ONLY be classified as a relevant complaint if it meets ALL these criteria:
            1. Explicitly expresses dissatisfaction or criticism about a system, policy, or public service
            2. The issue is under government control or regulation
            3. Suggests or implies a need for government action or policy change
            4. Addresses systemic or policy-level issues, not individual cases

            Do NOT classify as complaints if the post is:
            1. Merely asking questions or seeking information
            2. Requesting personal advice or help
            3. General discussion or opinion polls
            4. Sharing news or information
            5. Reporting individual incidents without policy implications
            6. Expressing gratitude or positive feedback

            Examples of TRUE (relevant complaints):
            - "Why can't Singapore ban single use plastics?" (Environmental policy criticism)
            - "Healthcare waiting times are unacceptably long" (Healthcare system issue)
            - "Public transport fares keep increasing but service declining" (Transport policy issue)
            - "HDB prices are becoming impossible for young couples" (Housing policy issue)
            - "Government's vaccination policies are too restrictive" (Healthcare policy issue)

            Examples of FALSE (not complaints):
            - "What documents do I need for passport renewal?" (Information seeking)
            - "Where can I report a noisy neighbor?" (Help seeking)
            - "Should we ban smoking in HDB?" (Discussion question)
            - "New COVID-19 cases reported today" (News sharing)
            - "Anyone else experiencing SingPass issues?" (Technical query)
            - "Thanks to all healthcare workers" (Appreciation post)

            When in Doubt:  
            - Be very strict - only classify as complaint if it clearly criticizes policy or system
            - If it's just a question or discussion, classify as "No"
            - Individual problems without policy implications should be "No"

            Always format your response as follows for each text:
            
            1. "Yes/No", "Domain Category", Confidence Score, Sentiment Score, Importance Level
            2. "Yes/No", "Domain Category", Confidence Score, Sentiment Score, Importance Level
            3. "Yes/No", "Domain Category", Confidence Score, Sentiment Score, Importance Level
            ...
            {num_entries - 1}. "Yes/No", "Domain Category", Confidence Score, Sentiment Score, Importance Level
            {num_entries}. "Yes/No", "Domain Category", Confidence Score, Sentiment Score, Importance Level

            IMPORTANT FORMAT RULES:
            1. Each response must be on a single line
            2. Exactly 5 components separated by commas
            3. No explanations or additional text between responses
            4. Values must be numbers between -1 and 1
            5. Category must be exactly one of: {categories_str}
            6. Yes/No must be in quotes
            7. Category must be in quotes
            
            Example format:
            1. "Yes", "Transport", 0.95, -0.8, 0.7
            2. "No", "Others", 0.8, -0.2, 0.3

            There are {num_entries} input contents, so you should return {num_entries} rows of output. 
            The first entry indicates whether the text is a useful complaint that may be potentially beneficial to the government.
            The second entry indicates the domain category of the complaint, it must be one of the following categories:
            {categories_str}
            The third entry is a confidence score between 0 and 1 indicating the confidence level of whether it is a complaint that may be potentially beneficial to the government.
            1 means that it is definitely a complaint, 0 means that it is definitely not a complaint.
            The fourth entry is the sentiment score ranging from -1 to 1, where -1 indicates a negative sentiment, 0 is neutral, and 1 is positive.
            The fifth entry is the importance level, meaning its potential impact on the government, and the urgency level, ranging from 0 to 1, where 1 indicates the highest importance.


        """
    }
    
    messages = [system_message, user_instruction] + [{"role": "user", "content": text} for text in batch_texts]
    
    # Make an API request for the current batch 
    completion = client.chat.completions.create(
        model="deepseek-r1-250120",
        messages=messages,
        stream=False,
        temperature=0.0
    )
    
    # Parse the response
    results = []
    lines = completion.choices[0].message.content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and explanatory text
        if not line or line.startswith('Let me') or line.startswith('Entries') or 'But need to' in line:
            continue
            
        try:
            # Remove any numbering at the start
            if '. ' in line:
                line = line.split('. ', 1)[1]
                
            # Split by comma and clean up each part
            parts = [part.strip().strip('"') for part in line.split(',')]
            
            if len(parts) == 5:  # Only process lines with correct format
                intent = parts[0].lower()
                domain = parts[1]
                confidence = float(parts[2])
                sentiment = float(parts[3])
                importance = float(parts[4])
                
                results.append((intent, domain, confidence, sentiment, importance))
            else:
                print(f"Skipping malformed line: {line}")
                
        except (ValueError, IndexError) as e:
            print(f"Error processing line: {line}")
            continue
    
    # If we didn't get enough valid results, pad with default values
    while len(results) < len(batch_texts):
        results.append(('no', 'Others', 0.0, 0.0, 0.0))
    
    # If we got too many results, truncate
    if len(results) > len(batch_texts):
        results = results[:len(batch_texts)]
    
    return results

# Add a function to estimate time remaining for API calls
def estimate_time_remaining_for_api(api_start_time, current_batch, total_batches):
    elapsed_time = time.time() - api_start_time
    if current_batch == 0:
        return "Calculating..."
    estimated_total_time = (elapsed_time / current_batch) * total_batches
    remaining_time = estimated_total_time - elapsed_time
    return time.strftime("%H:%M:%S", time.gmtime(remaining_time))

# Function to remove leading and trailing quotation marks
def remove_quotes(text):
    return text.strip('"')

def categorize_complaints(df=None, categories=None, input_csv=None, output_csv=None):
    # Start timing for the entire process
    start_time = time.time()
    
    # First, apply news filtering if the input is a CSV file
    if df is None and input_csv is not None:
        from news_filter import filter_news_posts
        df = filter_news_posts(input_csv=input_csv)
    elif df is None:
        raise ValueError("Either a DataFrame or an input CSV file must be provided.")
    
    print(f"Starting complaint categorization on {len(df)} non-news posts")
    
    # Count and print the number of completely empty rows
    empty_rows_count = df.isnull().all(axis=1).sum()
    print(f"Number of completely empty rows: {empty_rows_count}")
    
    print(f"Number of rows after preprocessing: {len(df)}")
    
    # Combine title and selftext if available
    if 'selftext' in df.columns:
        df['combined_text'] = df.apply(
            lambda row: f"{row['title']} {row['selftext']}" if pd.notna(row['selftext']) else row['title'],
            axis=1
        )
    else:
        df['combined_text'] = df['title']
    
    # Initialize lists to store categories
    intent_categories = []
    domain_categories = []
    confidence_scores = []
    sentiment_scores = []
    importance_levels = []
    
    # Process in batches using ThreadPoolExecutor
    batch_size = 50
    total_batches = (len(df) + batch_size - 1) // batch_size  # Calculate total number of batches
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_texts = df['combined_text'][i:i+batch_size]
            futures.append(executor.submit(process_batch, batch_texts, categories))
        
        for index, future in enumerate(futures):
            categories = future.result()
            for intent_category, domain_category, confidence, sentiment, importance in categories:
                intent_categories.append(intent_category)
                domain_categories.append(domain_category)
                confidence_scores.append(confidence)
                sentiment_scores.append(sentiment)
                importance_levels.append(importance)
            
            # Print a message every 200 posts processed
            if (index + 1) * batch_size % 200 == 0:
                print(f"Processed {(index + 1) * batch_size} posts")
    
    # Add categories to the DataFrame
    df['Intent Category'] = intent_categories
    df['Domain Category'] = domain_categories
    df['Confidence Score'] = confidence_scores
    df['Sentiment Score'] = sentiment_scores
    df['Importance Level'] = importance_levels
    
    # Apply the cleaning function to 'Intent Category' and 'Domain Category'
    if 'Intent Category' in df.columns:
        df['Intent Category'] = df['Intent Category'].apply(remove_quotes)

    if 'Domain Category' in df.columns:
        df['Domain Category'] = df['Domain Category'].apply(remove_quotes)
    
    # Write the results to the output CSV if provided
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    
    # End timing and print the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    return df
