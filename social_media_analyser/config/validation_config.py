from common_components.validators.not_empty_validator import NotEmptyValidator
from common_components.validators.min_length_validator import MinLengthValidator
from common_components.validators.profanity_validator import ProfanityFilterValidator
from common_components.validators.regex_validator import RegexValidator
from common_components.validators.url_validator import URLValidator

# Define the sequence of validators
VALIDATORS = [
    {"type": "NotEmptyValidator"},
    {"type": "MinLengthValidator", "params": {"min_length": 10}},
    {"type": "ProfanityFilterValidator"},
    {"type": "RegexValidator", "params": {"pattern": r"^[a-zA-Z0-9\s.,!?]+$"}},
    {"type": "URLValidator"}
]

# Map validator names to classes
VALIDATOR_MAP = {
    "NotEmptyValidator": NotEmptyValidator,
    "MinLengthValidator": MinLengthValidator,
    "ProfanityFilterValidator": ProfanityFilterValidator,
    "RegexValidator": RegexValidator,
    "URLValidator": URLValidator
}

def create_validation_chain():
    """Dynamically builds the validation chain from configuration."""
    previous = None
    first_validator = None

    for validator_cfg in VALIDATORS:
        validator_class = VALIDATOR_MAP[validator_cfg["type"]]
        params = validator_cfg.get("params", {})
        validator = validator_class(**params)

        if previous:
            previous.set_next(validator)
        else:
            first_validator = validator

        previous = validator

    return first_validator
