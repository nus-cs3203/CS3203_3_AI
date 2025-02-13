# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import pandas as pd
import time  # Import the time module

client = OpenAI(
  api_key="put your api key here"
)

def categorize_complaints_from_csv(input_csv, output_csv):
    # Start timing
    start_time = time.time()
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Remove completely empty rows
    df.dropna(how='all', inplace=True)
    
    # Combine 'title' and 'selftext' for each row
    df['combined_text'] = df.apply(lambda row: f"{row['title']} {row['selftext']}", axis=1)
    
    # Prepare the system message
    system_message = {
        "role": "system", 
        "content": "You are a helpful assistant categorizing complaints."
    }
    
    # Initialize lists to store categories
    intent_categories = []
    domain_categories = []
    
    # Process in batches
    batch_size = 20
    # print(df.head())
    # print(df.tail())
    # print(len(df))
    for i in range(0, len(df), batch_size):
        print(f"Processing batch starting at index {i}")
        batch_texts = df['combined_text'][i:i+batch_size]
        # print(batch_texts)
        
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
                
                Very important:
                1. Do not invent intent domain categories(eg, "Legal") or domain categories yourself, 
                2. Do not put one of the intent categories (eg. "General negative feelings but not direct complaints", 
                "General positive feelings/observations but not positive feedback") in the domain category column or vice versa, only use the ones provided.
                So the first entry is one of the following six intent category, 
                Intent Categories: "Direct Complaint", "General negative feelings but not direct complaints", "Positive Feedback", "General positive feelings/observations but not positive feedback", "Neutral Discussion", "Looking for Suggestions"
                and the second entry is one of the following 16 domain category.
                Domain Categories: "Healthcare", "Education", "Transportation", "Employment", "Environmental", "Public Safety", "Social Services", "Recreation", "Housing", "Food Services", "Infrastructure", "Retail", "Technology", "Financial", "Noise"
                There are {num_entries} rows, so you should return {num_entries} rows of output. There should be no space between each row. 
                Very important:
                1. Do not invent intent domain categories(eg, "Legal") or domain categories yourself, 
                2. Do not put one of the intent categories (eg. "General negative feelings but not direct complaints", 
                "General positive feelings/observations but not positive feedback") in the domain category column or vice versa, only use the ones provided.
            """
        }
        
        messages = [system_message, user_instruction] + [{"role": "user", "content": text} for text in batch_texts]
        
        # Make an API request for the current batch
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=False,
            temperature=0.0  # Set temperature to 0 for deterministic responses
        )
        
        # Parse the response
        response_text = completion.choices[0].message.content
        print(response_text)  # Debugging: print the response
        categories = [line for line in response_text.strip().split('\n') if line.strip()]
        
        # Validate response length
        if len(categories) != len(batch_texts):
            print(f"Warning: Expected {len(batch_texts)} responses, but got {len(categories)}")
        
        # Add categories to the lists
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
    
    # Write the updated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    
    # End timing and print the total time taken
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

# Example usage
categorize_complaints_from_csv('data/500rows_filtered.csv', 'data/500rows_categorized.csv')


