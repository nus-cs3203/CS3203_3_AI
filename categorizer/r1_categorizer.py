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
            Here are reddit posts from r/Singapore, and we want to determine whether it is a useful complaint that may be potentially beneficial 
            to the government. A useful complaint should highlight a specific issue that requires government intervention or policy change.
            Please be very strict about identifying useful complaints. Don't include any posts that are not useful to the government. 
            For example, "Condos are too expensive" suggests a need for action on housing affordability.
            "Why is the road so crowded" suggests a need for action on traffic congestion.
            "The public healthcare wait times are too long" suggests a need for improvements in healthcare services.
            "There are not enough public parks in the city" suggests a need for more recreational spaces.
            
            In contrast, posts like "I hate rainy days in Singapore!" or "Why can't weekends be longer?" 
            "The coffee at my local cafe is too bitter" or "I wish my favorite TV show aired more often" 
            Although these are negative statements, they are not useful complaints for the government.

            Always format your response as follows for each text:
            
            1. "Yes/No", "Domain Category", Confidence Score
            2. "Yes/No", "Domain Category", Confidence Score
            3. "Yes/No", "Domain Category", Confidence Score
            ...
            {num_entries - 1}. "Yes/No", "Domain Category", Confidence Score
            {num_entries}. "Yes/No", "Domain Category", Confidence Score

            The first entry indicates whether the text is a useful complaint that may be potentially beneficial to the government.
            The second entry indicates the domain category of the complaint, it must be one of the following categories:
            {categories_str}
            The third entry is a confidence score between 0 and 1 indicating the confidence level of whether it is a complaint.
            1 means that it is definitely a complaint, 0 means that it is definitely not a complaint.
            There are {num_entries} input contents, so you should return {num_entries} rows of output. There should be no space between each row, and
            each response starts with a new line.
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
    
    # Parse the response to include confidence scores
    categories_with_confidence = []
    for line in completion.choices[0].message.content.strip().split('\n'):
        if line.strip():
            parts = line.split(',')
            if len(parts) == 3:
                try:
                    # Extract the confidence score
                    confidence = float(parts[2].strip())
                    # Remove any numbering from the first part
                    intent = parts[0].strip().split(' ', 1)[-1]
                    categories_with_confidence.append((intent, parts[1].strip(), confidence))
                except ValueError:
                    print(f"Warning: Could not parse confidence score in line: {line}")
                    categories_with_confidence.append((parts[0].strip(), parts[1].strip(), 0.0))  # Default confidence
            else:
                print(f"Warning: Unexpected format in line: {line}")
                categories_with_confidence.append(("Unknown", "Unknown", 0.0))  # Default values for incorrect format
    
    # Validate response length
    if len(categories_with_confidence) != len(batch_texts):
        print(f"Warning: Expected {len(batch_texts)} responses, but got {len(categories_with_confidence)}")
        # Fill all entries with dummy data if there's a mismatch
        categories_with_confidence = [("Unknown", "Unknown", 0.0)] * len(batch_texts)
    
    # Return parsed categories with confidence scores
    return categories_with_confidence

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
    
    # Read the CSV file if df is not provided
    if df is None and input_csv is not None:
        df = pd.read_csv(input_csv, usecols=['title'])  # Adjusted to only use 'title' if 'selftext' is not available

    if df is None:
        raise ValueError("Either a DataFrame or an input CSV file must be provided.")
    
    # Limit to the first 100 lines
    df = df.head(100)
    
    # Count and print the number of completely empty rows
    empty_rows_count = df.isnull().all(axis=1).sum()
    print(f"Number of completely empty rows: {empty_rows_count}")
    
    print(f"Number of rows after preprocessing: {len(df)}")
    
    # Use only 'title' for combined_text if 'selftext' is not available
    df['combined_text'] = df['title'].fillna('')  # Use title directly if selftext is not available
    
    # Initialize lists to store categories
    intent_categories = []
    domain_categories = []
    confidence_scores = []
    
    # Process in batches using ThreadPoolExecutor
    batch_size = 40
    total_batches = (len(df) + batch_size - 1) // batch_size  # Calculate total number of batches
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_texts = df['combined_text'][i:i+batch_size]
            futures.append(executor.submit(process_batch, batch_texts, categories))
        
        for index, future in enumerate(futures):
            categories = future.result()
            for intent_category, domain_category, confidence in categories:
                intent_categories.append(intent_category)
                domain_categories.append(domain_category)
                confidence_scores.append(confidence)
    

    
    # Add categories to the DataFrame
    df['Intent Category'] = intent_categories
    df['Domain Category'] = domain_categories
    df['Confidence Score'] = confidence_scores
    
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
