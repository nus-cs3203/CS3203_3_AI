import pandas as pd
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.length_validator import LengthValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.text_validator.regex_validator import RegexValidator
from common_components.data_validator.validator_logger import ValidatorLogger

# Initialize logger
logger = ValidatorLogger()

# Load CSV file into DataFrame
df = pd.read_csv("files/sentiment_scored_2023_data.csv")

# Create validators and chain them
validator_chain = (
    NotEmptyValidator(["title", "selftext"], logger)  # Removed "column" if unnecessary
    .set_next(OnlyStringValidator(["title", "selftext"], logger=logger))  
    .set_next(LengthValidator({"selftext": (5, 100)}, logger=logger))  
    .set_next(RegexValidator(["title"], ["^[A-Za-z0-9\s]+$"], logger=logger))  # Dict instead of list for clarity
)

# Validate the DataFrame
result = validator_chain.validate(df)

# Print validation result
print(result)
