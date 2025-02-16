# Import Logger and validators
from common_components.data_validator.general_validators.not_empty_validator import NotEmptyValidator
from common_components.data_validator.text_validator.length_validator import LengthValidator
from common_components.data_validator.text_validator.only_string_validator import OnlyStringValidator
from common_components.data_validator.text_validator.regex_validator import RegexValidator
from common_components.data_validator.validator_logger import ValidatorLogger

# Initialize logger
logger = ValidatorLogger()

# Create validators and chain them
validator_chain = (
    NotEmptyValidator("data", logger)
    .set_next(OnlyStringValidator("data", logger))
    .set_next(LengthValidator("data", 5, 20, logger))
    .set_next(RegexValidator("data", r"^[A-Za-z0-9\s]+$", logger))
)

# Sample Data
requests = [
    {"data": "HelloWorld"},  # ✅ Pass
    {"data": ""},            # ❌ Fail (Empty string)
    {"data": 12345},         # ❌ Fail (Not a string)
    {"data": "A very very very long string"},  # ❌ Fail (Too long)
    {"data": "Hello@World!"}  # ❌ Fail (Invalid characters)
]

# Run validation
for req in requests:
    print(validator_chain.validate(req))
