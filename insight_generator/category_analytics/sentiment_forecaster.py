from insight_generator.base_decorator import InsightDecorator
from prophet import Prophet
import pandas as pd

class TopicSentimentForecastDecorator(InsightDecorator):
    def __init__(self, wrapped, historical_data, log_file="sentiment_forecasts.txt",
                 time_col="created_utc", sentiment_col="sentiment_title_selftext_polarity",
                 category_col="Domain Category", forecast_days=7):
        """
        Forecasts sentiment trends for topics using Prophet.

        Args:
        - wrapped: Base insight generator
        - historical_data: DataFrame with past sentiment scores
        - log_file: File to store sentiment forecasts
        - time_col: Timestamp column
        - sentiment_col: Sentiment score column
        - category_col: Column indicating topic/category
        - forecast_days: Days to forecast into the future
        """
        super().__init__(wrapped)
        self.historical_data = historical_data
        self.log_file = log_file
        self.time_col = time_col
        self.sentiment_col = sentiment_col
        self.category_col = category_col
        self.forecast_days = forecast_days
    
    def extract_insights(self, df):
        """Processes DataFrame and appends sentiment forecasts per category."""
        if self.category_col not in self.historical_data.columns:
            raise ValueError(f"Column '{self.category_col}' not found in historical_data!")

        sentiment_forecasts = {}
        log_lines = []
        forecast_data = []  # List to hold forecast data for the DataFrame

        categories = df[self.category_col].unique()
        for category in categories:
            forecast_score, forecast_label, conf_interval = self.forecast_sentiment(category)

            sentiment_forecasts[category] = {
                "score": forecast_score,
                "label": forecast_label,
                "confidence_interval": conf_interval
            }

            # Prepare log entry
            log_lines.append(f"\nCategory: {category}, Score: {forecast_score:.4f}, Label: {forecast_label}")
            log_lines.append(f"Confidence Interval: {conf_interval}")

            # Sample posts
            sample_texts = df[df[self.category_col] == category]["selftext"].head(3)
            log_lines.append("Sample Posts:")
            log_lines.extend([f"- {text}" for text in sample_texts])

            # Append data for the DataFrame
            forecast_data.append({
                "category": category,
                "sentiment_predicted": forecast_label,
                "sentiment_score_predicted": forecast_score
            })

        # Write logs in batch for efficiency
        with open(self.log_file, "w") as log_file:
            log_file.write("\n".join(log_lines))

        # Convert forecast data to DataFrame and return
        forecast_df = pd.DataFrame(forecast_data)
        return forecast_df

    def forecast_sentiment(self, category):
        """Uses Prophet to forecast sentiment for a given category."""
        category_data = self.historical_data[self.historical_data[self.category_col] == category]

        if category_data.empty or self.sentiment_col not in category_data.columns:
            return 0.0, "neutral", (0.0, 0.0)  # Default if no data

        # Convert time column to datetime
        category_data = category_data[[self.time_col, self.sentiment_col]].copy()
        category_data["ds"] = pd.to_datetime(category_data[self.time_col], errors="coerce", utc=True).dt.tz_localize(None)
        category_data["y"] = category_data[self.sentiment_col]

        # Ensure enough data points before fitting
        if len(category_data) < 10:
            return 0.0, "neutral", (0.0, 0.0)  # Skip if too few data points

        # Aggregate to weekly level if data is sparse
        if category_data["ds"].diff().dt.days.median() > 7:
            category_data = category_data.resample("W", on="ds").mean().reset_index()

        # Dynamic forecast period based on data length
        forecast_days = min(len(category_data) // 2, self.forecast_days)

        # Train Prophet model
        model = Prophet()
        model.fit(category_data[["ds", "y"]])

        # Predict future sentiment
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Extract last forecasted sentiment and confidence interval
        forecasted_sentiment = forecast["yhat"].iloc[-1]
        forecasted_sentiment = max(-1, min(forecasted_sentiment, 1))
        conf_interval = (forecast["yhat_lower"].iloc[-1], forecast["yhat_upper"].iloc[-1])

        # Classify sentiment
        if forecasted_sentiment > 0.1:
            sentiment_label = "positive"
        elif forecasted_sentiment < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return forecasted_sentiment, sentiment_label, conf_interval
