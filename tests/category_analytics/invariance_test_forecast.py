import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from insight_generator.base_insight import BaseInsightGenerator
from insight_generator.category_analytics.sentiment_forecaster import TopicSentimentForecastDecorator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
original_df = pd.read_csv("tests/category_analytics/data/raw_invariance_test_forecast.csv")
modified_df = pd.read_csv("tests/category_analytics/data/modified_invariance_test_forecast.csv")

# Define tolerance levels
TOLERANCE = 0.1  # Allowable deviation in forecasted sentiment scores

base_generator = BaseInsightGenerator()
forecast_decorator = TopicSentimentForecastDecorator(base_generator)
original_df = forecast_decorator.extract_insights(original_df)
modified_df = forecast_decorator.extract_insights(modified_df)

# Merge the dataframes on category to compare forecasts
merged_df = original_df.merge(modified_df, on="category", suffixes=("_orig", "_mod"))

# Initialize test results
results = []

# Create plot for forecasting comparison
plt.figure(figsize=(10, 6))

for _, row in merged_df.iterrows():
    category = row["category"]
    forecast_orig = row["forecasted_sentiment_orig"]
    forecast_mod = row["forecasted_sentiment_mod"]
    
    # Check forecast stability
    deviation = abs(forecast_orig - forecast_mod)
    stable = deviation <= TOLERANCE
    
    # Check missing/insufficient data handling
    valid_data_handling = not (np.isnan(forecast_orig) or np.isnan(forecast_mod))
    
    # Store results
    results.append({
        "category": category,
        "forecast_stable": stable,
        "valid_data_handling": valid_data_handling
    })
    
    # Plot forecasted sentiment comparison
    plt.plot([category, category], [forecast_orig, forecast_mod], marker="o", label=category)
    
    # Log warnings if any test fails
    if not stable:
        logger.warning(f"Category {category}: Forecast deviation {deviation} exceeds tolerance {TOLERANCE}")
    if not valid_data_handling:
        logger.warning(f"Category {category}: Missing or insufficient data detected")

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("tests/category_analytics/data/invariance_test_results_forecast.csv", index=False)

logger.info("Invariance test completed. Results saved to invariance_test_results.csv")

# Show the plot
plt.xlabel('Category')
plt.ylabel('Forecasted Sentiment')
plt.title('Forecasted Sentiment Comparison')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
