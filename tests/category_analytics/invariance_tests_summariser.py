import pandas as pd
from sentence_transformers import SentenceTransformer, util
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.llm_category_summarizer import CategorySummarizerDecorator
import matplotlib.pyplot as plt

# Load original and modified datasets
original_df = pd.read_csv("tests/sentiment_analyser/data/raw_invariance_test_sentiment.csv") 
modified_df = pd.read_csv("tests/category_analytics/data/perturbed_invariance_test_sentiment.csv")

# Initialize summarizer
base_generator = BaseInsightGenerator()
summarizer = CategorySummarizerDecorator(base_generator)

# Generate summaries
original_insights = summarizer.extract_insights(original_df)
modified_insights = summarizer.extract_insights(modified_df)

# Merge summaries by category
merged_df = original_insights.merge(modified_insights, on="category", suffixes=("_orig", "_mod"))

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute semantic similarity
similarities = []
for _, row in merged_df.iterrows():
    orig_summary = row["summary_orig"]
    mod_summary = row["summary_mod"]
    
    orig_embedding = model.encode(orig_summary, convert_to_tensor=True)
    mod_embedding = model.encode(mod_summary, convert_to_tensor=True)
    
    similarity = util.pytorch_cos_sim(orig_embedding, mod_embedding).item()
    similarities.append(similarity)

# Add similarity scores to DataFrame
merged_df["semantic_similarity"] = similarities

# Save results
merged_df.to_csv("tests/category_analytics/data/semantic_similarity_results.csv", index=False)

print("Semantic similarity results saved.")

sem_res = pd.read_csv("tests/category_analytics/data/semantic_similarity_results.csv")

# Set a threshold value
threshold = 0.7

# Plot semantic similarity
plt.figure(figsize=(10, 6))
plt.bar(sem_res["category"], sem_res["semantic_similarity"], color="skyblue", label="Semantic Similarity")

# Plot threshold line
plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")

# Add labels and title
plt.xlabel("Category")
plt.ylabel("Semantic Similarity")
plt.title("Semantic Similarity by Category")
plt.xticks(rotation=45, ha="right")
plt.legend()

# Save plot to file
plt.tight_layout()
plt.savefig("tests/category_analytics/data/semantic_similarity_plot.png")

# Create a grouped bar chart for better comparison
categories = sem_res["category"]
x = range(len(categories))

# Plot original and modified summaries' similarities
plt.figure(figsize=(12, 6))
plt.bar(x, sem_res["semantic_similarity"], color="skyblue", label="Semantic Similarity")

# Add threshold line
plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")

# Add labels and title
plt.xticks(x, categories, rotation=45, ha="right")
plt.xlabel("Category")
plt.ylabel("Semantic Similarity")
plt.title("Semantic Similarity by Category")
plt.legend()

# Save grouped bar chart to file
plt.tight_layout()
plt.savefig("tests/category_analytics/data/semantic_similarity_comparison_plot.png")
plt.show()