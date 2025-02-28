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
            to the government. For example, Condos are too expensive may suggest that the government should take action to address the issue of housing affordability.
            Why is the road so crowded may suggest that the government should take action to address the issue of traffic congestion.
            In contrast, life with newborn baby in SG, missing ns vocation deadline are not useful complaints for the government.

            Always format your response as follows for each text:
            
            1. "Yes/No", "Domain Category"
            2. "Yes/No", "Domain Category"
            3. "Yes/No", "Domain Category"
            ...
            {num_entries - 1}. "Yes/No", "Domain Category"
            {num_entries}. "Yes/No", "Domain Category"

            So the first entry means whether the text is a useful complaint that may be potentially beneficial to the government.
            The second entry means the domain category of the complaint, it must be one of the following categories:
            {categories_str}
            There are {num_entries} input contents, so you should return {num_entries} rows of output. There should be no space between each row, and
            each response starts with a new line.
        """
    }
    
    messages = [system_message, user_instruction] + [{"role": "user", "content": text} for text in batch_texts]
    
    #print(messages)

    # Make an API request for the current batch 
    completion = client.chat.completions.create(
        model="deepseek-v3-241226",
        messages=messages,
        stream=False,
        temperature=0.0
    )
    
    # Parse the response
    response_text = completion.choices[0].message.content
    #print(response_text)
    categories = [line for line in response_text.strip().split('\n') if line.strip()]
    
    # Validate response length
    if len(categories) != len(batch_texts):
        print(f"Warning: Expected {len(batch_texts)} responses, but got {len(categories)}")
        # Fill all entries with dummy data if there's a mismatch
        categories = ['"Unknown", "Unknown"'] * len(batch_texts)
    
    # Return parsed categories
    return categories

def categorize_complaints(df=None, categories=None, input_csv=None, output_csv=None):
    # Start timing
    start_time = time.time()
    
    # Read the CSV file if df is not provided
    if df is None and input_csv is not None:
        df = pd.read_csv(input_csv, usecols=['title'])  # Adjusted to only use 'title' if 'selftext' is not available

    if df is None:
        raise ValueError("Either a DataFrame or an input CSV file must be provided.")
    
    # Count and print the number of completely empty rows
    empty_rows_count = df.isnull().all(axis=1).sum()
    print(f"Number of completely empty rows: {empty_rows_count}")
    
    print(f"Number of rows after preprocessing: {len(df)}")
    
    # Optionally, remove completely empty rows
    # df.dropna(how='all', inplace=True)
    
    # Use only 'title' for combined_text if 'selftext' is not available
    df['combined_text'] = df['title'].fillna('')  # Use title directly if selftext is not available
    
    # Initialize lists to store categories
    intent_categories = []
    domain_categories = []
    
    # Process in batches using ThreadPoolExecutor
    batch_size = 20
    # Use ThreadPoolExecutor without specifying max_workers
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_texts = df['combined_text'][i:i+batch_size]
            futures.append(executor.submit(process_batch, batch_texts, categories))
        
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

    print(f"Length of intent_categories: {len(intent_categories)}")
    print(f"Length of domain_categories: {len(domain_categories)}")

    df['Intent Category'] = intent_categories
    df['Domain Category'] = domain_categories
    
    # Write the results to the output CSV if provided
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    
    # End timing and print the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    return df
# Example usage
#categorize_complaints('data/2023_filtered_data.csv', 'data/2023_categorized_chunked2.csv')
