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
    You will be given the title of a Reddit post. Your job is to classify whether the post describes a complaint that is relevant to government authorities.

Guidelines:
- A complaint is a statement that something is unsatisfactory or unacceptable, where the person is expressing some dissatisfaction or encouraging some form of action to be taken.
- A relevant complaint should be an opinion that the government authorities can act on to improve the country. If the complaint is about a particular brand or product, it is not relevant to the authority. The complaint should be related to a high-level issue, such as transport or healthcare that the government is responsible for.
- Questions that are simply seeking discussions are not complaints.

Output:
Output true if it is a relevant complaint, else output false.

Examples:
- Protect The Unvax Singaporeans From Losing Jobs -> true (Encouraging job security within the country)
- How do I get user flair in this sub? -> false (Unrelated to problems in the country))
-  Why are central 24/7 clinics so infamously bad? What are your negative or even positive experiences with them? -> true (Mentioned about clinics being bad)
-  Should Smoking At Home Be Banned?  -> false (General discussion)
- Singpass app keeps crashing -> true (Dissatisfaction at the app)
-  Singapore's toxic overwork culture has got to go -> true (Dissatisfaction at the culture)
-  NS missed out my enlistment. Need help -> false (This is asking for help and not complaining about a problem) 
- PLEASE BE CAREFUL OF THIS NEW SCAM! -> false (This is not a complaint, it is just advice given) 
- What's going to happen to all the folk involved in the Safe Entry business?  -> false (This is just a question asking for input, and it is not expressing any dissatisfaction)


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
        #model="deepseek-r1-250120",
        model='deepseek-v3-250324',
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

def categorize_complaints(df=None, categories=None, input_csv=None, output_csv=None, batch_size=50, is_second_round=False):
    """
    Parameters batch_size and is_second_round control the batch processing size and verification round
    """
    # Use smaller batch size for second round verification
    if is_second_round:
        batch_size = 10
    
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
    total_batches = (len(df) + batch_size - 1) // batch_size
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
