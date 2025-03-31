import pandas as pd
from sentence_transformers import SentenceTransformer, util
from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.llm_category_absa import CategoryABSAWithLLMInsightDecorator
import matplotlib.pyplot as plt

# Load original and modified datasets
original_df = pd.read_csv("tests/category_analytics/files/raw_data_for_post_train.csv")
modified_df = pd.read_csv("tests/category_analytics/files/modified_invariance_test.csv")

# Initialize summarizer
base_generator = BaseInsightGenerator()
absa = CategoryABSAWithLLMInsightDecorator(base_generator)

# Generate summaries
original_insights = absa.extract_insights(original_df)
modified_insights = absa.extract_insights(modified_df)

# Merge summaries by category
merged_df = original_insights.merge(modified_insights, on="category", suffixes=("_orig", "_mod"))

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

################################# KEYWORDS ###########################
def compute_similarity(list1, list2):
    """Compute semantic similarity between two lists of keywords."""
    str1 = " ".join(list1) if isinstance(list1, list) else str(list1)
    str2 = " ".join(list2) if isinstance(list2, list) else str(list2)
    
    orig_embedding = model.encode(str1, convert_to_tensor=True)
    mod_embedding = model.encode(str2, convert_to_tensor=True)
    
    return util.pytorch_cos_sim(orig_embedding, mod_embedding).item()

# Compute semantic similarity for keywords
merged_df["semantic_similarity_keywords"] = merged_df.apply(
    lambda row: compute_similarity(row["keywords_orig"], row["keywords_mod"]), axis=1
)

################################# ASPECTS ###########################
# Compute semantic similarity for ABSA results
merged_df["semantic_similarity_absa_result"] = merged_df.apply(
    lambda row: compute_similarity(row["absa_result_orig"], row["absa_result_mod"]), axis=1
)

# Save results
merged_df.to_csv("tests/category_analytics/files/semantic_similarity_results_absa.csv", index=False)
print("Semantic similarity results saved.")

# Load results for plotting
sem_res = pd.read_csv("tests/category_analytics/files/semantic_similarity_results_absa.csv")

# Set a threshold value
threshold = 0.7

# Plot semantic similarity
plt.figure(figsize=(10, 6))
plt.bar(sem_res["category"], sem_res["semantic_similarity_keywords"], color="skyblue", label="Semantic Similarity")

# Plot threshold line
plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")

# Add labels and title
plt.xlabel("Category")
plt.ylabel("Semantic Similarity")
plt.title("Semantic Similarity by Category")
plt.xticks(rotation=45, ha="right")
plt.legend()

# Show plot
plt.tight_layout()
plt.show()