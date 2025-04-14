import random
import re
import pandas as pd

df = pd.read_csv("tests/sentiment_analyser/data/raw_invariance_test_sentiment.csv")

# Simple lists of words/phrases to replace or append
replacement_dict = {
    "great": "terrible",
    "good": "bad",
    "helpful": "useless",
    "interesting": "boring",
    "important": "pointless",
    "success": "failure",
    "positive": "negative",
    "improve": "worsen",
    "better": "worse",
    "safe": "dangerous",
    "support": "criticize",
    "hope": "doubt",
    "strong": "weak"
}

negative_suffixes = [
    " This ended up being a complete failure.",
    " Unfortunately, it was a waste of time.",
    " What a disappointment.",
    " Things only got worse from there.",
    " It caused more harm than good."
]

def perturb_text(text):
    if pd.isna(text) or text.strip().lower() in ["[removed]", ""]:
        return text

    # Lowercase the text for easier replacement
    modified = text

    # Apply replacements
    for pos_word, neg_word in replacement_dict.items():
        pattern = re.compile(rf"\b{pos_word}\b", flags=re.IGNORECASE)
        modified = pattern.sub(neg_word, modified)

    # Add a negative suffix randomly
    if random.random() < 0.7:
        modified += random.choice(negative_suffixes)

    return modified

# Apply perturbations
df["title"] = df["title"].apply(perturb_text)
df["description"] = df["description"].apply(perturb_text)

# Save the perturbed version
df.to_csv("tests/sentiment_analyser/data/directional_sentiment_test_data.csv", index=False)
