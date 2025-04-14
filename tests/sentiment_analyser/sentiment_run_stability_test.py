import pandas as pd
import matplotlib.pyplot as plt

from common_components.data_preprocessor.concrete_general_builder import GeneralPreprocessorBuilder
from common_components.data_preprocessor.director import PreprocessingDirector
from sentiment_analyser.context import SentimentAnalysisContext
from sentiment_analyser.polarity.custom import CustomSentimentClassifier
from sentiment_analyser.polarity.advanced import AdvancedSentimentClassifier

# Load dataset
df = pd.read_csv("tests/sentiment_analyser/data/raw_invariance_test_sentiment.csv").head(100)
df['title_with_desc'] = df['title'] + " " + df['description']

# Define critical and text columns
CRITICAL_COLUMNS = ["title"]
TEXT_COLUMNS = ["title_with_desc"]
SUBSET_COLUMNS = ["title", "description"]

# Step 1: Preprocessing
builder = GeneralPreprocessorBuilder(critical_columns=CRITICAL_COLUMNS, text_columns=TEXT_COLUMNS, data=df, subset=SUBSET_COLUMNS)
director = PreprocessingDirector(builder)
director.construct_builder()
df = builder.get_result()

# List of classifiers to test
classifiers = [
    ("Custom", CustomSentimentClassifier()),
    ("Advanced", AdvancedSentimentClassifier()),
]

# Initialize results dictionaries to store sentiment scores and labels for plotting
all_results_scores = {name: [] for name, _ in classifiers}
all_results_labels = {name: [] for name, _ in classifiers}

# Run polarity classifiers one by one with clear output
for name, classifier in classifiers:
    print(f"\n===== Running {name} Sentiment Analysis =====")
    context = SentimentAnalysisContext(classifier)
    
    # Run the analysis 3 times to calculate sentiment deviations
    runs_scores = []
    runs_labels = []
    for i in range(3):
        df_copy = df.copy()
        df_copy = context.analyze(df_copy, text_cols=["title_with_desc"])
        
        # Get sentiment scores and labels
        sentiment_scores = df_copy['title_with_desc_score'].tolist()
        sentiment_labels = df_copy['title_with_desc_label'].tolist()
        
        runs_scores.append(sentiment_scores)
        runs_labels.append(sentiment_labels)
    
    # Calculate sentiment deviations for scores
    sentiment_deviations = [abs(runs_scores[i][j] - runs_scores[i+1][j]) for i in range(2) for j in range(len(runs_scores[0]))]
    avg_deviation = sum(sentiment_deviations) / len(sentiment_deviations)
    print(f"Average Sentiment Deviation for {name} over 3 runs: {avg_deviation}")
    
    # Store the sentiment scores and labels for plotting
    all_results_scores[name] = runs_scores
    all_results_labels[name] = runs_labels

# Create separate figures for each classifier
for name, runs in all_results_scores.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f"{name} Sentiment Score Distribution Across 3 Runs", fontsize=16)
    
    for run_idx, (ax, run) in enumerate(zip(axes, runs)):
        ax.hist(run, bins=20, alpha=0.7, density=True, color='blue')
        ax.set_title(f"Run {run_idx+1}")
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Density')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title
    plt.savefig(f"tests/sentiment_analyser/data/{name.lower()}_sentiment_score_distribution_3_runs.png")
    plt.show()

# Plot the sentiment labels distribution across the runs for each classifier
plt.figure(figsize=(10, 6))
fig, axes = plt.subplots(len(classifiers), 1, figsize=(10, 6 * len(classifiers)), sharex=True)

for ax, (name, runs) in zip(axes, all_results_labels.items()):
    for run_idx, run in enumerate(runs):
        # Count the occurrences of each label in the run
        label_counts = {label: run.count(label) for label in ['positive', 'neutral', 'negative']}
        labels = list(label_counts.keys())
        counts = list(label_counts.values())
        
        ax.bar(labels, counts, alpha=0.5, label=f"Run {run_idx+1}")
    
    ax.set_title(f"{name} Sentiment Label Distribution Across 3 Runs")
    ax.set_ylabel('Count')
    ax.legend(loc='best')

axes[-1].set_xlabel('Sentiment Label')
plt.tight_layout()
plt.show()
plt.savefig("tests/sentiment_analyser/data/sentiment_label_distribution_3_runs.png")
