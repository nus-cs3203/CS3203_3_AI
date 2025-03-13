from insight_generator.base_decorator import InsightDecorator
from prophet import Prophet
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicSentimentForecastDecorator(InsightDecorator):
    def __init__(self, wrapped, historical_data=None, log_file="sentiment_forecasts.txt",
                 time_col="date", sentiment_col="sentiment",
                 category_col="domain_category", forecast_months=1):
        super().__init__(wrapped)
        self.log_file = log_file
        self.time_col = time_col
        self.sentiment_col = sentiment_col
        self.category_col = category_col
        self.forecast_months = forecast_months  # Forecast next month
        
        # Load historical data if not provided
        self.historical_data = historical_data if historical_data is not None else pd.read_csv("files/historical_data_for_training.csv")
    
    def extract_insights(self, df):
        if self.category_col not in df.columns:
            raise ValueError(f"Column '{self.category_col}' not found in input dataframe!")

        sentiment_forecasts = []
        log_lines = []

        # Get unique categories from the input dataframe
        categories = df[self.category_col].unique()
        for category in categories:
            try:
                forecast_score, conf_interval = self.forecast_sentiment(df, category)
                sentiment_forecasts.append({
                    "domain_category": category,
                    "forecasted_sentiment": forecast_score
                })

                log_lines.append(f"\nCategory: {category}, Forecasted Score: {forecast_score:.4f}")
                log_lines.append(f"Confidence Interval: {conf_interval}")
            except Exception as e:
                logger.error(f"Error forecasting sentiment for category {category}: {e}")
                sentiment_forecasts.append({
                    "domain_category": category,
                    "forecasted_sentiment": 0.0
                })
                log_lines.append(f"\nCategory: {category}, Error: {e}")

        # Save logs to the specified log file
        with open(self.log_file, "w") as log_file:
            log_file.write("\n".join(log_lines))

        return pd.DataFrame(sentiment_forecasts)
    
    def forecast_sentiment(self, df, category):
        # Filter the dataframe for the specific category
        # Ensure category comparison is case-insensitive and strips any leading/trailing spaces
        category_data = df[df[self.category_col].str.strip().str.lower() == str(category).strip().lower()].copy()

        if category_data.empty:
            logger.warning(f"No data for category: {category}")
            return 0.0, (0.0, 0.0)

        # Check if the sentiment column exists in the filtered data
        if self.sentiment_col not in category_data.columns:
            logger.warning(f"Sentiment column '{self.sentiment_col}' not found for category: {category}")
            return 0.0, (0.0, 0.0)

        # Check if there are any null values in the time or sentiment columns
        category_data = category_data.dropna(subset=[self.time_col, self.sentiment_col])

        if category_data.empty:
            logger.warning(f"Data after cleaning is empty for category: {category}")
            return 0.0, (0.0, 0.0)

        # Convert time column to datetime and prepare for Prophet
        category_data["ds"] = pd.to_datetime(category_data[self.time_col], errors="coerce", utc=True).dt.tz_localize(None)

        # Ensure the sentiment column is numeric
        try:
            category_data["y"] = pd.to_numeric(category_data[self.sentiment_col], errors="coerce")
        except Exception as e:
            logger.error(f"Error converting sentiment column to numeric for category {category}: {e}")
            return 0.0, (0.0, 0.0)

        # Drop rows with NaN sentiment values after conversion
        category_data = category_data.dropna(subset=["y"])

        if category_data.empty:
            logger.warning(f"Data after converting sentiment to numeric is empty for category: {category}")
            return 0.0, (0.0, 0.0)

        # Ensure there are at least 2 data points
        if len(category_data) < 2:
            logger.warning(f"Not enough data for category: {category} (less than 10 data points)")
            return 0.0, (0.0, 0.0)

        # Resample data to monthly frequency
        try:
            # Ensure only numeric columns are included in resampling
            category_data = category_data[["ds", "y"]].resample("ME", on="ds").mean().reset_index()
        except Exception as e:
            logger.error(f"Error resampling data for category {category}: {e}")
            return 0.0, (0.0, 0.0)

        # Check if the resampled data is empty after resampling
        if category_data.empty:
            logger.warning(f"Data is empty after resampling for category: {category}")
            return 0.0, (0.0, 0.0)

        # Fit the Prophet model
        model = Prophet()
        model.fit(category_data[["ds", "y"]])

        # Forecast for the next `forecast_months` months
        future = model.make_future_dataframe(periods=self.forecast_months, freq='ME')
        forecast = model.predict(future)

        # Forecasted sentiment score (clipped between -1 and 1)
        forecasted_sentiment = max(-1, min(forecast["yhat"].iloc[-1], 1))
        conf_interval = (forecast["yhat_lower"].iloc[-1], forecast["yhat_upper"].iloc[-1])

        return forecasted_sentiment, conf_interval