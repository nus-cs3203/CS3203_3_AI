from openai import OpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import os

client = OpenAI(
    api_key="af598933-afc1-4095-853b-c00879ec86d0",
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

def process_opinion_batch(batch_texts):
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant identifying personal opinions and emotional expressions in Reddit posts."
    }
    
    num_entries = len(batch_texts)
    
    user_instruction = {
        "role": "user",
        "content": """
            You will be given Reddit post titles. Your job is to identify posts that express personal opinions, emotions, or subjective experiences.
            Output 'true' if the post contains personal opinions, emotions, or subjective views, output 'false' for neutral descriptions or objective information.

            Should output 'true' (Personal/Subjective content):
            - Singapore's healthcare system needs improvement (Strong opinion)
            - Why is the COE price so ridiculous? (Emotional question)
            - I'm tired of the rising cost of living here (Personal frustration)
            - MRT breakdown again, this is unacceptable! (Complaint)
            - The new regulations are too strict (Opinion)
            - Really disappointed with the service at polyclinic (Personal experience + emotion)
            - Housing prices are getting out of hand (Opinion + frustration)
            - Can't believe how crowded the buses are (Complaint)
            
            Should output 'false' (Neutral/Objective content):
            - Police appeal for information on missing girl
            - MOH reports 842 new Covid-19 cases today
            - New BTO launch in Woodlands announced
            - Looking for recommendations for good restaurants
            - Where to buy computer parts in Singapore?
            - Bus service 123 route changed from next month
            - Anyone knows the opening hours of this shop?
            - What documents needed for passport renewal
            
            Format your response as follows for each text, one per line:
            1. true/false
            2. true/false
            ...

            There are {num_entries} input contents, so you should return {num_entries} lines.
            """.format(num_entries=num_entries)
    }
    
    messages = [system_message, user_instruction] + [{"role": "user", "content": text} for text in batch_texts]
    
    completion = client.chat.completions.create(
        model="deepseek-r1-250120",
        messages=messages,
        stream=False,
        temperature=0.0
    )
    
    # Parse the response
    results = []
    for line in completion.choices[0].message.content.strip().split('\n'):
        if line.strip():
            # Remove any numbering and get just the true/false value
            has_opinion = line.strip().split('.')[-1].strip().lower() == 'true'
            results.append(has_opinion)
    
    # Validate response length
    if len(results) != len(batch_texts):
        print(f"Warning: Expected {len(batch_texts)} responses, but got {len(results)}")
        results = [False] * len(batch_texts)
    
    return results

def is_media_source_url(url):
    """Check if the URL is from a media source"""
    if not isinstance(url, str):
        return False
        
    media_domains = [
        # Major media outlets
        'straitstimes.com',
        'channelnewsasia.com',
        'todayonline.com',
        'mothership.sg',
        'zaobao.com',
        'businesstimes.com.sg',
        'tnp.sg',
        'bloomberg.com',
        'reuters.com',
        'yahoo.com/news',
        'asiaone.com',
        
        # News aggregators and other sites
        'mustsharenews.com',
        'vulcanpost.com',
        '8world.com',
        'scmp.com',
        'theindependent.sg',
        'theonlinecitizen.com',
        'ricemedia.co',
        'stomp.straitstimes.com',
        'zula.sg',
        'coconuts.co',
        'theedgesingapore.com',
        'thekopi.co',
        '8days.sg',
        'taiwannews.com.tw',
        'vice.com',
        'cnn.com',
        'firstpost.com',
        'thestar.com.my',
        'finance.yahoo.com',
        'planb.sg',
        
        # Media subdomains and shortcuts
        'sg.news.yahoo.com',
        'edition.cnn.com',
        'str.sg',
        'tdy.sg',
        
        # Additional media sources
        'asiaabc.news',
        'subbiznet.co.uk',
        'theregister.com',
        'techandpublicgood.com',
        'crackfact.com',
        'voltearena.com',
        'kuanyewism.com',
        'sgplus.one',
        
        # Video platforms
        'youtube.com/watch',
        'youtu.be',
        'tiktok.com',
        'streamable.com'
    ]
    
    # Reddit and image hosting domains
    reddit_domains = [
        'reddit.com/r/',
        'redd.it',
        'i.redd.it',
        'v.redd.it',
        'reddit.com/gallery',
        'i.imgur.com'
    ]
    
    # Excluded domains
    excluded_domains = [
        'routetofi.blogspot.com',
        'starlightinternational786.world',
        'onlinegdb.com',
        'cryptotabbrowser.com',
        'creatoronline.org'
    ]
    
    # Keep reddit and image posts for analysis
    if any(domain in url.lower() for domain in reddit_domains):
        return False
        
    # Exclude specific domains
    if any(domain in url.lower() for domain in excluded_domains):
        return False
        
    return any(domain in url.lower() for domain in media_domains)

def filter_for_opinions(df=None, input_csv=None, output_folder=None):
    start_time = time.time()
    
    if df is None and input_csv is not None:
        df = pd.read_csv(input_csv)

    if df is None:
        raise ValueError("Either a DataFrame or an input CSV file must be provided.")
    
    initial_count = len(df)
    print(f"Initial number of rows: {initial_count}")
    
    # First preprocessing: Remove deleted/removed posts
    deleted_patterns = ['[deleted by user]', '[deleted]', '[removed]']
    mask_deleted = ~df['title'].isin(deleted_patterns)
    df_filtered = df[mask_deleted].copy()
    
    deleted_count = initial_count - len(df_filtered)
    print(f"Removed {deleted_count} deleted/removed posts")
    print(f"Remaining posts after removing deleted content: {len(df_filtered)}")
    
    # Second filter: Remove posts from media websites
    if 'url' in df_filtered.columns:
        df_filtered['is_media_url'] = df_filtered['url'].apply(is_media_source_url)
        media_posts = df_filtered['is_media_url'].sum()
        print(f"Found {media_posts} posts from media websites (filtered out)")
        
        # Save media source posts for verification
        df_filtered[df_filtered['is_media_url']].to_csv(os.path.join(output_folder, "media_source_posts.csv"), index=False)
        
        # Keep non-media posts
        df_filtered = df_filtered[~df_filtered['is_media_url']].copy()
        df_filtered.to_csv(os.path.join(output_folder, "posts_for_analysis.csv"), index=False)
        
        print(f"Remaining posts after URL filtering: {len(df_filtered)}")
        
        # Save statistics
        stats = {
            'Initial posts': initial_count,
            'Deleted/removed posts': deleted_count,
            'Media source posts': media_posts,
            'Posts for opinion analysis': len(df_filtered),
            'Media source percentage': f"{(media_posts/(initial_count-deleted_count))*100:.2f}%"
        }
        
        with open(os.path.join(output_folder, "filter_stats.txt"), 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    else:
        print("No URL column found in the CSV file")
        
    # Combine title and selftext for content analysis
    if 'selftext' in df_filtered.columns:
        df_filtered['combined_text'] = df_filtered.apply(
            lambda row: f"{row['title']} {row['selftext']}" if pd.notna(row['selftext']) else row['title'],
            axis=1
        )
    else:
        df_filtered['combined_text'] = df_filtered['title']
    
    # Process posts in batches using ThreadPoolExecutor
    batch_size = 50
    has_opinion_list = []
    
    print("Starting opinion analysis for remaining posts...")
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df_filtered), batch_size):
            batch_texts = df_filtered['combined_text'][i:i+batch_size]
            futures.append(executor.submit(process_opinion_batch, batch_texts))
        
        for index, future in enumerate(futures):
            batch_results = future.result()
            has_opinion_list.extend(batch_results)
            
            if (index + 1) * batch_size % 200 == 0:
                print(f"Analyzed {(index + 1) * batch_size} posts")
    
    # Add the content-based classification to the DataFrame
    df_filtered['has_opinion'] = has_opinion_list
    opinions_found = sum(has_opinion_list)
    print(f"Found {opinions_found} posts containing personal opinions/emotions")
    
    # Keep only posts with personal opinions
    df_final = df_filtered[df_filtered['has_opinion']].copy()
    
    final_count = len(df_final)
    print(f"Final number of opinion posts: {final_count}")
    print(f"Total posts filtered out: {initial_count - final_count}")
    
    # Write the results to the output CSV if provided
    if output_folder is not None:
        output_path = os.path.join(output_folder, "opinion_posts.csv")
        df_final.to_csv(output_path, index=False)
    
    end_time = time.time()
    print(f"Total time taken: {(end_time - start_time):.2f} seconds")
    return df_final

if __name__ == "__main__":
    # Set input and output paths
    input_file = "last_round_files/all_posts_2022_2025.csv"
    output_folder = "last_round_files/opinion_filter_test_1000"
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Read first 1000 rows of data
    df = pd.read_csv(input_file, nrows=1000)
    
    # Run the complete filtering process
    filtered_df = filter_for_opinions(
        df=df,
        output_folder=output_folder
    )
    
    print("Opinion filtering complete!") 