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
            
            1. "Yes/No", "Domain Category"
            2. "Yes/No", "Domain Category"
            3. "Yes/No", "Domain Category"
            ...
            {num_entries - 1}. "Yes/No", "Domain Category"
            {num_entries}. "Yes/No", "Domain Category"

            The first entry indicates whether the text is a useful complaint that may be potentially beneficial to the government.
            The second entry indicates the domain category of the complaint, it must be one of the following categories:
            {categories_str}
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
    
    # Parse the response
    response_text = completion.choices[0].message.content
    categories = [line for line in response_text.strip().split('\n') if line.strip()]
    
    # Validate response length
    if len(categories) != len(batch_texts):
        print(f"Warning: Expected {len(batch_texts)} responses, but got {len(categories)}")
        # Fill all entries with dummy data if there's a mismatch
        categories = ['"Unknown", "Unknown"'] * len(batch_texts)
    
    # Return parsed categories
    return categories

# Add a function to estimate time remaining for API calls
def estimate_time_remaining_for_api(api_start_time, current_batch, total_batches):
    elapsed_time = time.time() - api_start_time
    if current_batch == 0:
        return "Calculating..."
    estimated_total_time = (elapsed_time / current_batch) * total_batches
    remaining_time = estimated_total_time - elapsed_time
    return time.strftime("%H:%M:%S", time.gmtime(remaining_time))

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
    explanations = []
    
    # Process in batches using ThreadPoolExecutor
    batch_size = 10
    total_batches = (len(df) + batch_size - 1) // batch_size  # Calculate total number of batches
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_texts = df['combined_text'][i:i+batch_size]
            futures.append(executor.submit(process_batch, batch_texts, categories))
        
        for index, future in enumerate(futures):
            categories = future.result()
            for category in categories:
                # Remove numbering and split by comma
                parts = category.split('. ', 1)[-1].split(',')
                if len(parts) >= 2:
                    intent_categories.append(parts[0].strip('" '))
                    domain_categories.append(parts[1].strip('" '))
                else:
                    # Handle unexpected format by appending default values
                    print(f"Unexpected format in category: {category}")
                    intent_categories.append("Unknown")
                    domain_categories.append("Unknown")
    
    # Add categories to the DataFrame
    df['Intent Category'] = intent_categories
    df['Domain Category'] = domain_categories
    df['Explanation'] = explanations
    
    # Write the results to the output CSV if provided
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    
    # End timing and print the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    return df

# Example usage
# categorize_complaints(input_csv='csv_results/all_complaints_2022_2025_strict_cleaned.csv', output_csv='csv_results/all_complaints_2022_2025_stricter_100.csv') 