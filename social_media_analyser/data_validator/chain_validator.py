from social_media_analyser.config.validation_config import create_validation_chain

class ChainValidator:
    def __init__(self):
        self.chain = create_validation_chain()

    def validate(self, data):
        """Runs the validation chain."""
        return self.chain.validate(data)
