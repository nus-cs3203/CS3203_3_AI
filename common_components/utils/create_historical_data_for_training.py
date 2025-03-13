import pandas as pd

# Read the CSV file
file_path = "/Users/aishwaryahariharaniyer/Desktop/CS3203_3_AI/files/2022_2025_merged.csv"
data = pd.read_csv(file_path)

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Filter data for the years 2022 to 2023
filtered_data = data[(data['date'].dt.year >= 2022) & (data['date'].dt.year <= 2024)]

# Balance the data based on sentiment score
positive_data = filtered_data[filtered_data['sentiment'] > 0]
negative_data = filtered_data[filtered_data['sentiment'] < 0]
neutral_data = filtered_data[filtered_data['sentiment'] == 0]

# Find the minimum length among the three categories
min_length = min(len(positive_data), len(negative_data), len(neutral_data))

# Sample the data to balance it
positive_data = positive_data.sample(min_length)
negative_data = negative_data.sample(min_length)
neutral_data = neutral_data.sample(min_length)

# Concatenate the balanced data
balanced_data = pd.concat([positive_data, negative_data, neutral_data])

# Shuffle the data
filtered_data = balanced_data.sample(frac=1).reset_index(drop=True)

# Display the filtered data
filtered_data.to_csv("files/historical_data_for_training.csv", index=False)