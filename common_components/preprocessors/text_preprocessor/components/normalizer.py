import re

class Normalizer:
    def process(self, text):
        if isinstance(text, list):
            return [self._normalize_string(t) for t in text]
        elif isinstance(text, str):
            return self._normalize_string(text)
        else:
            raise ValueError("Input should be a string or a list of strings")

    def _normalize_string(self, text):
        text = text.lower().strip()
        text = re.sub(r'[:;xX8=][-~]?[)DPO]', '<smiley>', text)  # Replace common emoticons
        text = re.sub(r'[:;][-~]?(\()', '<sad>', text)  # Replace sad emoticons
        return text
