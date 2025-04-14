import re
import pandas as pd

df = pd.read_csv("tests/sentiment_analyser/data/raw_invariance_test_sentiment.csv")
# Define perturbation mappings
gender_map = {
    r'\bhe\b': 'she',
    r'\bshe\b': 'he',
    r'\bhis\b': 'her',
    r'\bher\b': 'his',
    r'\bman\b': 'woman',
    r'\bwoman\b': 'man',
    r'\bmen\b': 'women',
    r'\bwomen\b': 'men',
}

country_map = {
    'Singapore': 'Malaysia',
    'Iran': 'Malaysia',
    'Malaysia': 'Singapore',
    'China': 'India',
    'India': 'China',
    'USA': 'Russia',
    'Russia': 'USA',
}

sg_place_map = {
    'Orchard': 'Yishun',
    'Bukit Timah': 'Jurong',
    'Holland': 'Woodlands',
    'Sentosa': 'Hougang',
    'Yishun': 'Orchard',
    'Jurong': 'Bukit Timah',
    'Woodlands': 'Holland',
    'Hougang': 'Sentosa',
}

race_map = {
    'Chinese': 'Malay',
    'Malay': 'Indian',
    'Indian': 'Chinese',
    'Eurasian': 'Peranakan',
    'Peranakan': 'Eurasian',
}

# Combine all maps
combined_maps = [gender_map, country_map, sg_place_map, race_map]

# Function to apply perturbations and note changes
def perturb_text(text, maps):
    changes = []
    original = text
    if pd.isna(text):
        return text, changes
    for mapping in maps:
        for k, v in mapping.items():
            pattern = re.compile(k, flags=re.IGNORECASE)
            if re.search(pattern, text):
                changes.append(f"{re.findall(pattern, text)} → {v}")
                text = pattern.sub(lambda m: preserve_case(m.group(), v), text)
    return text, changes

# Preserve case helper
def preserve_case(original, replacement):
    if original.isupper():
        return replacement.upper()
    elif original[0].isupper():
        return replacement.capitalize()
    else:
        return replacement

# Apply to both 'title' and 'description'
perturbed_df = df.copy()
perturbed_df['perturbed_title'], title_changes = zip(*df['title'].apply(lambda x: perturb_text(x, combined_maps)))
perturbed_df['perturbed_description'], desc_changes = zip(*df['description'].apply(lambda x: perturb_text(x, combined_maps)))

# Save the change logs for traceability
perturbed_df['title_changes'] = title_changes
perturbed_df['description_changes'] = desc_changes

# Show a sample
perturbed_df[['title', 'perturbed_title', 'title_changes', 'description', 'perturbed_description', 'description_changes']].head()
perturbed_df.to_csv("tests/sentiment_analyser/data/perturbed_invariance_test_sentiment.csv", index=False)