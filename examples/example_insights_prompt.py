import pandas as pd
from insight_generator.poll_generator import PollGenerator

# Sample Reddit posts
df = pd.read_csv("files/all_complaints_2022_2025.csv").head(100)  # Load complaints data

# Apply decorator
prompt_decorator = PollGenerator()
insights = prompt_decorator.extract_insights(df)
print(insights.head(5))
insights.to_csv("files/poll_prompts_20225.csv", index=False)
