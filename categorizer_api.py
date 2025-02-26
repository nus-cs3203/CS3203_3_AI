from openai import OpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

client = OpenAI(
    api_key="af598933-afc1-4095-853b-c00879ec86d0",
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

def process_batch(batch_texts):
    """Process a batch of texts for categorization"""
    system_message = {
        "role": "system", 
        "content": "You are a helpful assistant categorizing complaints."
    }
    
    num_entries = len(batch_texts)
    user_instruction = {
        "role": "user", 
        "content": f"""
            Here are reddit posts from r/Singapore, and we want to determine:
            1. Whether it is a useful complaint that may be potentially beneficial to the government
            2. What domain it belongs to

            Always format your response as follows for each text:
            
            1. "Yes/No", "Domain Category"
            2. "Yes/No", "Domain Category"
            ...
            {num_entries}. "Yes/No", "Domain Category"

            For the first part:
            - "Yes" if it's a useful complaint for the government
            - "No" if it's not a useful complaint

            The Domain Category must be one of:
            Housing, Healthcare, Transportation, Public Safety, Transport, Education, Environment, Employment, 
            Public Health, Legal, Economy, Politics, Technology, Infrastructure, Others
        """
    }
    
    messages = [system_message, user_instruction] + [{"role": "user", "content": text} for text in batch_texts]
    
    completion = client.chat.completions.create(
        model="deepseek-v3-241226",
        messages=messages,
        stream=False,
        temperature=0.0
    )
    
    response_text = completion.choices[0].message.content
    categories = [line for line in response_text.strip().split('\n') if line.strip()]
    
    if len(categories) != len(batch_texts):
        categories = ['"No", "Others"'] * len(batch_texts)
    
    return categories

def categorize_for_api(df):
    """
    Categorize complaints from DataFrame for API use
    
    Args:
        df: pandas DataFrame containing posts
    Returns:
        DataFrame with Intent and Domain categories
    """
    df['combined_text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
    
    intent_categories = []
    domain_categories = []
    
    batch_size = 20
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_texts = df['combined_text'][i:i+batch_size]
            futures.append(executor.submit(process_batch, batch_texts))
        
        for future in futures:
            categories = future.result()
            for category in categories:
                parts = category.split('. ', 1)[-1].split(',')
                if len(parts) == 2:
                    intent_categories.append(parts[0].strip('" '))
                    domain_categories.append(parts[1].strip('" '))
                else:
                    intent_categories.append("No")
                    domain_categories.append("Others")
    
    return pd.DataFrame({
        'Intent Category': intent_categories,
        'Domain Category': domain_categories
    }) 