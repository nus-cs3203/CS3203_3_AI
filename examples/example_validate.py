import pandas as pd
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.length_validator import LengthValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.text_validator.regex_validator import RegexValidator
from common_components.data_validator.validator_logger import ValidatorLogger

logger = ValidatorLogger()  # Initialize logger

df = pd.read_csv("files/sentiment_scored_2023_data.csv")  # Load CSV into DataFrame

# Create and chain validators
validator_chain = (
    NotEmptyValidator(["title", "selftext"], logger)
    .set_next(OnlyStringValidator(["title", "selftext"], logger=logger))
    .set_next(LengthValidator({"selftext": (5, 100)}, logger=logger))
)

result = validator_chain.validate(df)  # Validate DataFrame

print(result)  # Print result
