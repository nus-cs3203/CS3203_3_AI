import pandas as pd
import matplotlib.pyplot as plt
from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.polarity.advanced import AdvancedSentimentClassifier

# Load original and modified datasets
original_df = pd.read_csv("tests/sentiment_analyser/data/raw_invariance_test_sentiment.csv")

modified_df = pd.read_csv("tests/sentiment_analyser/data/perturbed_invariance_test_sentiment.csv")
modified_df.drop(columns=["title", "description"], inplace=True, errors='ignore')
modified_df.rename(columns={"perturbed_title": "title", "perturbed_description": "description"}, inplace=True)

# Define critical and text columns
CRITICAL_COLUMNS = ["title"]
TEXT_COLUMNS = ["title_with_desc"]
SUBSET_COLUMNS = ["title", "description"]

# Step 1: Preprocessing for both original and modified datasets
def preprocess_data(df):
    builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df, subset=SUBSET_COLUMNS)
    director = PreprocessingDirector(builder)
    director.construct_builder()
    return builder.get_result()

original_df = preprocess_data(original_df)
modified_df = preprocess_data(modified_df)

# List of classifiers to test
classifiers = [
    ("Advanced", AdvancedSentimentClassifier()),
]

# Initialize results dictionaries to store sentiment scores
all_results_scores = {name: {"original": [], "modified": []} for name, _ in classifiers}

# Initialize category-level trends (if needed for further analysis)
category_trends = {name: [] for name, _ in classifiers}

# Set a threshold value for comparison
threshold = 0.7

# Run polarity classifiers one by one
for name, classifier in classifiers:
    print(f"\n===== Running {name} Sentiment Analysis =====")
    
    context = SentimentAnalysisContext(classifier)
    
    # Analyze sentiment for both original and modified datasets
    original_copy = original_df.copy()
    modified_copy = modified_df.copy()

    original_copy = context.analyze(original_copy, text_cols=["title_with_desc"])
    modified_copy = context.analyze(modified_copy, text_cols=["title_with_desc"])
    
    # Get sentiment scores for both datasets
    original_scores = original_copy['title_with_desc_score']
    modified_scores = modified_copy['title_with_desc_score']
    
    # Store sentiment scores for plotting
    all_results_scores[name]["original"] = original_scores
    all_results_scores[name]["modified"] = modified_scores
    
    # Check if the sentiment scores fall within a reasonable threshold
    score_agreement = (abs(original_scores - modified_scores) <= threshold).mean()
    print(f"Agreement on sentiment scores for {name}: {score_agreement:.2f}")

# --- Plotting and Report Generation ---

# Plot percentage alignment of sentiment scores and save the plots
for name, scores in all_results_scores.items():
    plt.figure(figsize=(10, 6))
    
    # Calculate percentage alignment per post
    alignment = (abs(scores["original"] - scores["modified"]) <= threshold) * 100
    
    # Plot percentage alignment
    plt.bar(range(len(alignment)), alignment, alpha=0.7, label=f"{name} Alignment (%)")
    
    plt.xlabel('Sample Index')
    plt.ylabel('Percentage Alignment')
    plt.title(f'Sentiment Score Alignment: {name}')
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot to file
    plt.savefig(f"tests/sentiment_analyser/data/{name}_sentiment_alignment_plot.png")
    plt.close()

# --- Generate Text Report ---
report = []

# Report on sentiment score agreement
report.append("===== Sentiment Score Comparison Report =====")
for name, scores in all_results_scores.items():
    score_agreement = (abs(scores["original"] - scores["modified"]) <= threshold).mean()
    report.append(f"{name} Agreement on Sentiment Scores: {score_agreement:.2f}")

# Save the report to a text file
with open("tests/sentiment_analyser/data/sentiment_analysis_comparison_report.txt", "w") as file:
    file.write("\n".join(report))

print("Sentiment analysis comparison report and plots saved.")
