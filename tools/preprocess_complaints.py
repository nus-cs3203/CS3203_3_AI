import pandas as pd
import os


def preprocess_complaints(file_paths: list) -> pd.DataFrame:
    """
    Preprocess the complaints data by replacing 'created_utc' with a formatted date
    and removing extra columns after 'view count'.

    :param file_paths: List of paths to the input CSV files.
    :return: Combined preprocessed DataFrame.
    """
    # Specify the columns to read
    columns_to_read = [
        'author_flair_text', 'downs', 'likes', 'name', 'no_follow', 'num_comments',
        'score', 'selftext', 'title', 'ups', 'upvote_ratio', 'url', 'view_count', 'created_utc'
    ]

    # Initialize an empty DataFrame
    combined_df = pd.DataFrame()

    for file_path in file_paths:
        # Read the CSV file with specified columns
        df = pd.read_csv(file_path, usecols=columns_to_read, low_memory=False)

        # Remove extra columns after 'view_count'
        if 'view_count' in df.columns:
            view_count_index = df.columns.get_loc('view_count')
            df = df.iloc[:, :view_count_index + 1]

        # Convert 'created_utc' to numeric, coercing errors to NaN
        if 'created_utc' in df.columns:
            df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
            df['date'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
            df.drop(columns=['created_utc'], inplace=True)

        # Filter posts starting from 2022-01-01
        df = df[df['date'] >= '2022-01-01']

        # Append to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Debugging line to print the DataFrame after preprocessing
    print("Combined DataFrame after preprocessing:", combined_df)

    return combined_df


if __name__ == "__main__":
    # Example usage
    folder_path = "last_round_files"
    file_names = [
        "filtered_singapore_submission.csv",
        "filtered_singapore_submissions_with_comments_2024.csv",
        "filtered_singapore_submissions_with_comments_2025_complete.csv"
    ]
    file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]
    preprocessed_df = preprocess_complaints(file_paths)
    # Save the preprocessed DataFrame to a new CSV file
    preprocessed_df.to_csv(os.path.join(folder_path, "all_posts_2022_2025.csv"), index=False)
    print("Preprocessing completed successfully.")
    print("Preprocessed data saved to last_round_files/all_posts_2022_2025.csv") 