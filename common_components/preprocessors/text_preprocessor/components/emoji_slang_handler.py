import emoji

class EmojiSlangHandler:
    def __init__(self):
        self.slang_dict = {
            "brb": "be right back",
            "lol": "laugh out loud",
            "omg": "oh my god",
            "idk": "i don't know",
            "btw": "by the way",
        }

    def process(self, words):
        words = [self.slang_dict.get(word, word) for word in words]  # Replace slang
        words = [emoji.demojize(word) if word in emoji.UNICODE_EMOJI["en"] else word for word in words]  # Convert emojis to text
        return words
