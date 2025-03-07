import pandas as pd
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.general_validators.numeric_range_validator import NumericRangeValidator
from common_components.data_validator.text_validator.length_validator import LengthValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.text_validator.regex_validator import RegexValidator
from common_components.data_validator.validator_logger import ValidatorLogger

# Define columns for processing
CRITICAL_COLUMNS = ["title", "sentiment_comments_polarity", "selftext", "comments", "sentiment_title_selftext_polarity"]
TEXT_COLUMNS = ["title", "selftext", "comments"]
SUBSET_COLUMNS = ["title", "selftext"]
NUMERIC_COLUMNS = ["sentiment_comments_polarity", "sentiment_title_selftext_polarity"]

logger = ValidatorLogger()

# Initialize validators
v1 = NotEmptyValidator(CRITICAL_COLUMNS, logger)
v2 = OnlyStringValidator(TEXT_COLUMNS, logger)
v3 = RegexValidator(TEXT_COLUMNS, [r"^(?!.*%%%).*"] * len(TEXT_COLUMNS), logger)
v4 = NumericRangeValidator(NUMERIC_COLUMNS, logger, min_value=-1, max_value=1)
v5 = LengthValidator({col: (10, 1000) for col in SUBSET_COLUMNS}, logger)

# Set up chain of responsibility
v1.set_next(v2).set_next(v3).set_next(v4).set_next(v5)

#INVALID DATA (missing values, text that is too long)
data1 = pd.read_csv("tests/preprocessor_validator/data/input.csv") #has missing values, text that is too long

# INVALID DATA (text that is too long, title is not a string)
data2 = data1.copy().dropna(subset=CRITICAL_COLUMNS)
# Modify the last 2 rows and change the title
data2.iloc[-2:, data2.columns.get_loc("title")] = [0, 1]

# INVALID DATA (text that is too long, text contains %%%)
data3 = data1.copy().dropna(subset=CRITICAL_COLUMNS)
data3.iloc[-2:, data3.columns.get_loc("title")] = ["title", "title%%%"]

# INVALID DATA (text that is too long, score is not in range)
data4 = data1.copy().dropna(subset=CRITICAL_COLUMNS)
data4.iloc[-2:, data4.columns.get_loc("sentiment_comments_polarity")] = [-101, 101]

# INVALID DATA (text that is too long)
data5 = data1.copy().dropna(subset=CRITICAL_COLUMNS)

# VALID DATA
data = data1.copy().dropna(subset=CRITICAL_COLUMNS)
for col in TEXT_COLUMNS:
    data = data[data[col].str.len() < 1000]
    data = data[data[col].str.len() > 10]


def test_v1_fail():
    try:
        v1.validate(data1)
        print("Test failed: No error was raised for invalid data1.")
    except ValueError as e:
        print(f"Test passed: Caught expected ValueError for data1 -> {e}")
    except Exception as e:
        print(f"Test failed: Unexpected error type for data1 -> {e}")

def test_v2_fail():
    try:
        v1.validate(data2)
        print("Test failed: No error was raised for invalid data2.")
    except ValueError as e:
        print(f"Test passed: Caught expected ValueError for data2 -> {e}")
    except Exception as e:
        print(f"Test failed: Unexpected error type for data2 -> {e}")

def test_v3_fail():
    try:
        v1.validate(data3)
        print("Test failed: No error was raised for invalid data3.")
    except ValueError as e:
        print(f"Test passed: Caught expected ValueError for data3 -> {e}")
    except Exception as e:
        print(f"Test failed: Unexpected error type for data3 -> {e}")

def test_v4_fail():
    try:
        v1.validate(data4)
        print("Test failed: No error was raised for invalid data4.")
    except ValueError as e:
        print(f"Test passed: Caught expected ValueError for data4 -> {e}")
    except Exception as e:
        print(f"Test failed: Unexpected error type for data4 -> {e}")

def test_v5_fail():
    try:
        v1.validate(data5)
        print("Test failed: No error was raised for invalid data5.")
    except ValueError as e:
        print(f"Test passed: Caught expected ValueError for data5 -> {e}")
    except Exception as e:
        print(f"Test failed: Unexpected error type for data5 -> {e}")

def test_valid_data():
    try:
        v1.validate(data)
        print("Test passed: No error was raised for valid data.")
    except Exception as e:
        print(f"Test failed: Unexpected error for valid data -> {e}")

if __name__ == "__main__":
    test_v1_fail()
    test_v2_fail()
    test_v3_fail()
    test_v4_fail()
    test_v5_fail()
    test_valid_data()


