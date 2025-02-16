from openai import OpenAI
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor

client = OpenAI(
    api_key="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"
)

def process_batch(batch_texts):
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
            Based on the following reddit posts, categorize each into one of the intent categories and also categorize it into one of the domain categories.
            Always format your response as follows for each text:
            
            1. "Intent Category", "Domain Category"
            2. "Intent Category", "Domain Category"
            ...
            {num_entries}. "Intent Category", "Domain Category"
            
            The first entry is one of the following six intent categories, 
            Intent Categories: "Direct Complaint", "General negative feelings but not direct complaints", "Positive Feedback", "General positive feelings/observations but not positive feedback", "Neutral Discussion", "Looking for Suggestions"
            and the second entry is one of the following 16 domain categories.
            Domain Categories: "Healthcare", "Education", "Transportation", "Employment", "Environmental", "Public Safety", "Social Services", "Recreation", "Housing", "Food Services", "Infrastructure", "Retail", "Technology", "Financial", "Noise"
            There are {num_entries} rows, so you should return {num_entries} rows of output. There should be no space between each row. 
            Please note a very important rule:
            1. For the first entry intent category, only choose one between "Direct Complaint", "General negative feelings but not direct complaints", "Positive Feedback", "General positive feelings/observations but not positive feedback", "Neutral Discussion", "Looking for Suggestions"
            2. For the second entry domain category, only choose one between "Healthcare", "Education", "Transportation", "Employment", "Environmental", "Public Safety", "Social Services", "Recreation", "Housing", "Food Services", "Infrastructure", "Retail", "Technology", "Financial", "Noise"
        """
    }
    
    messages = [system_message, user_instruction] + [{"role": "user", "content": text} for text in batch_texts]
    
    # Make an API request for the current batch
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=False,
        temperature=0.0
    )
    
    # Parse the response
    response_text = completion.choices[0].message.content
    print(response_text)
    categories = [line for line in response_text.strip().split('\n') if line.strip()]
    
    # Validate response length
    if len(categories) != len(batch_texts):
        print(f"Warning: Expected {len(batch_texts)} responses, but got {len(categories)}")
        # Fill all entries with dummy data if there's a mismatch
        categories = ['"Unknown", "Unknown"'] * len(batch_texts)
    
    # Return parsed categories
    return categories

def categorize_complaints(input_csv, output_csv):
    # Start timing
    start_time = time.time()
    
    # Read the entire CSV file
    df = pd.read_csv(input_csv, usecols=[0, 1])
    
    # Count and print the number of completely empty rows
    empty_rows_count = df.isnull().all(axis=1).sum()
    print(f"Number of completely empty rows: {empty_rows_count}")
    
    # Optionally, remove completely empty rows
    # df.dropna(how='all', inplace=True)
    
    # Combine 'title' and 'selftext' for each row
    df['combined_text'] = df.apply(lambda row: f"{row['title']} {row['selftext']}", axis=1)
    
    # Initialize lists to store categories
    intent_categories = []
    domain_categories = []
    
    # Process in batches using ThreadPoolExecutor
    batch_size = 20
    # Set the maximum number of threads
    max_threads = 5  # Adjust this number based on your needs and system capabilities
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_texts = df['combined_text'][i:i+batch_size]
            futures.append(executor.submit(process_batch, batch_texts))
        
        for future in futures:
            categories = future.result()
            for category in categories:
                # Remove numbering and split by comma
                parts = category.split('. ', 1)[-1].split(',')
                if len(parts) == 2:
                    intent_categories.append(parts[0].strip('" '))
                    domain_categories.append(parts[1].strip('" '))
                else:
                    # Handle unexpected format
                    print(f"Unexpected format in category: {category}")
    
    # Add categories to the DataFrame
    df['Intent Category'] = intent_categories
    df['Domain Category'] = domain_categories
    
    # Write the results to the output CSV
    df.to_csv(output_csv, index=False)
    
    # End timing and print the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

# Example usage for processing the entire file
categorize_complaints('data/2023_filtered_data.csv', 'data/2023_categorized_chunked.csv') 