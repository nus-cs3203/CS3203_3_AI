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

# Initialize results dictionaries to store sentiment scores for plotting
all_results_scores = {name: [] for name, _ in classifiers}

# Initialize category-level results for trend analysis
category_trends = {name: [] for name, _ in classifiers}
time_trends = {name: [] for name, _ in classifiers}

# Run polarity classifiers one by one with clear output
for name, classifier in classifiers:
    print(f"\n===== Running {name} Sentiment Analysis =====")
    context = SentimentAnalysisContext(classifier)
    
    # Analyze sentiment and store the trends
    df_copy = df.copy()
    df_copy = context.analyze(df_copy, text_cols=["title_with_desc"])
    
    # Get sentiment scores
    sentiment_scores = df_copy['title_with_desc_score']
    
    # Calculate category-level sentiment trends (group by category)
    category_trend = df_copy.groupby('category')['title_with_desc_score'].mean()  # Average sentiment score per category
    category_trends[name] = category_trend
    
    # Track sentiment trends over time (assuming you have a date or order-based column like 'date')
    df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')  # Ensure 'date' is in datetime format
    time_trend = df_copy.groupby(df_copy['date'].dt.date)['title_with_desc_score'].mean()  # Average sentiment per day
    time_trends[name] = time_trend
    
    # Store the sentiment scores for plotting
    all_results_scores[name] = sentiment_scores

# --- Trend Analysis: Category-level and Time-level ---
# Save and show Category-level Sentiment Trend
for name, category_trend in category_trends.items():
    plt.figure(figsize=(10, 6))
    category_trend.plot(kind='bar', label=f"{name} Sentiment by Category")
    plt.xlabel('Category')
    plt.ylabel('Average Sentiment Score')
    plt.title(f'{name} Sentiment Trend by Category')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{name}_category_trend.png")  # Save the plot
    plt.show()

# Save and show Time-level Sentiment Trend (assumes 'date' exists in dataset)
for name, time_trend in time_trends.items():
    plt.figure(figsize=(10, 6))
    time_trend.plot(label=f"{name} Sentiment Over Time")
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.title(f'{name} Sentiment Trend Over Time')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{name}_time_trend.png")  # Save the plot
    plt.show()

# --- Generate Text Report ---
report = []

# Report on category trends
report.append("===== Sentiment Trends by Category =====")
for name, category_trend in category_trends.items():
    report.append(f"{name} Sentiment Trend by Category:\n{category_trend}\n")

# Report on time trends
report.append("===== Sentiment Trends Over Time =====")
for name, time_trend in time_trends.items():
    report.append(f"{name} Sentiment Trend Over Time:\n{time_trend}\n")

# Save report to a text file
with open("sentiment_analysis_report.txt", "w") as file:
    file.write("\n".join(report))

print("Report generated: sentiment_analysis_report.txt")
