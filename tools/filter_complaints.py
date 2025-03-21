import pandas as pd

def filter_complaints(input_file: str, output_file: str):
    """
    Filter complaints with sentiment score lower than -0.5 and confidence score greater than 0.8.

    :param input_file: Path to the input CSV file.
    :param output_file: Path to save the filtered CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter the DataFrame
    filtered_df = df[(df['sentiment'] < -0.5) & (df['confidence'] > 0.8)]
    
    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    input_file = "last_round_files/all_complaints_2022_2025.csv"
    output_file = "last_round_files/filtered_complaints.csv"
    filter_complaints(input_file, output_file) 