import re

class Normalizer:
    def normalize(self, text):
        text = text.lower().strip()
        text = re.sub(r'[:;xX8=][-~]?[)DPO]', '<smiley>', text)  # Replace common emoticons
        text = re.sub(r'[:;][-~]?(\()', '<sad>', text)  # Replace sad emoticons
        return text
