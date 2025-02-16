import pandas as pd

def analyze_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Total number of posts
    total_posts = len(df)

    # Number of complaints
    complaints = df[df['Intent Category'] == 'Direct Complaint']
    num_complaints = len(complaints)

    # Number of complaints by domain category
    domain_complaint_counts = complaints['Domain Category'].value_counts()

    # Number of posts by intent category
    intent_counts = df['Intent Category'].value_counts()

    # Monthly complaint counts for each category in 2023
    df['created_utc'] = pd.to_datetime(df['created_utc'])
    df_2023 = df[df['created_utc'].dt.year == 2023]
    monthly_complaints = df_2023[df_2023['Intent Category'] == 'Direct Complaint']
    monthly_trends = monthly_complaints.groupby([monthly_complaints['created_utc'].dt.to_period('M'), 'Domain Category']).size().unstack(fill_value=0)

    # Print results
    print(f"Total posts: {total_posts}")
    print(f"Total complaints: {num_complaints}")
    print("\nNumber of complaints by domain category:")
    print(domain_complaint_counts)
    print("\nNumber of posts by intent category:")
    print(intent_counts)
    print("\nMonthly complaint counts for each category in 2023:")
    print(monthly_trends)

# Example usage
analyze_data('data/sentiment_scored_2023_data.csv')