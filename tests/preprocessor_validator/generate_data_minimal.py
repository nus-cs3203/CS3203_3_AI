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

# -------------------------
# Now, create the expected data by applying the pipeline transformations
# -------------------------

# Load the test data
df_expected = pd.read_csv("tests/preprocessor_validator/data/input.csv")
print("Check expected", df_expected.shape)

# Ensure 'created_utc' is in datetime format
df_expected["created_utc"] = pd.to_datetime(df_expected["created_utc"], errors="coerce", utc=True)

# Remove duplicate rows based on critical columns and drop rows with missing critical values
df_expected.drop_duplicates(subset=CRITICAL_COLUMNS, inplace=True)
print("Dropping duplicates", df_expected.shape)

df_expected.dropna(subset=CRITICAL_COLUMNS, inplace=True)
# Replace NaN values in numeric columns with 0
numeric_cols = df_expected.select_dtypes(include=[np.number]).columns
df_expected[numeric_cols] = df_expected[numeric_cols].fillna(0)
print("Handle missing values", df_expected.shape)

# Join the 'title' and 'selftext' columns into a new column 'title_with_desc'
df_expected['title_with_desc'] = df_expected[SUBSET_COLUMNS].astype(str).agg(' '.join, axis=1)
print("Joining cols", df_expected.shape)

df_expected.reset_index(drop=True, inplace=True)

# Save the expected data output
df_expected.to_csv("tests/preprocessor_validator/data/expected_minimal.csv", index=False)

print("Test and expected data have been saved successfully.")
