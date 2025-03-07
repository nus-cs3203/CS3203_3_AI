import pandas as pd
import numpy as np

# Read the original data
df = pd.read_csv("files/sentiment_scored_2023_data.csv")
print("Starting with", df.shape)

# Ensure 'created_utc' is in datetime format
df["created_utc"] = pd.to_datetime(df["created_utc"], errors="coerce", utc=True)

# Define columns for processing
CRITICAL_COLUMNS = ["title"]
TEXT_COLUMNS = ["title", "selftext", "comments"]
SUBSET_COLUMNS = ["title", "selftext"]

# Set a reproducible seed for consistent sampling
np.random.seed(42)
frac = 0.4  # 40% modification for each transformation

# Modification 1: Randomly set 40% of 'title' values to NaN
nan_title_indices = df.sample(frac=frac, random_state=42).index
df.loc[nan_title_indices, 'title'] = np.nan
print("Modification 1", df.shape)

# Modification 2: Randomly set 40% of 'selftext' values to NaN
nan_selftext_indices = df.sample(frac=frac, random_state=43).index
df.loc[nan_selftext_indices, 'selftext'] = np.nan
print("Modification 2", df.shape)

# Modification 3: Randomly add 20% copies
df = pd.concat([df, df.sample(frac=0.2, random_state=44)])
print("Modification 3", df.shape)

# Save the modified test data for your pipeline tests
df.to_csv("tests/preprocessor_validator/data/input.csv", index=False)

