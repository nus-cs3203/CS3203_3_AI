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
        insights = self._wrapped.extract_insights(df)

        # Open log file once for efficiency
        with open(self.log_file, "w") as log_file:
            sentiment_forecasts = {}

            for category, group in df.groupby(self.category_col):
                forecast_score, forecast_label = self.forecast_sentiment(category)
                
                sentiment_forecasts[category] = {
                    "score": forecast_score,
                    "label": forecast_label
                }

                # Log forecasted sentiment + sample text from each category
                log_file.write(f"\nCategory: {category}, Score: {forecast_score:.4f}, Label: {forecast_label}\n")
                log_file.write("Sample Posts:\n")
                for text in group["selftext"].head(3):  # Log first 3 posts
                    log_file.write(f"- {text}\n")

        insights["sentiment_forecasts"] = sentiment_forecasts
        return insights

    def forecast_sentiment(self, category):
        """Uses Prophet to forecast sentiment for a given category."""
        category_data = self.historical_data[self.historical_data[self.category_col] == category]

        if category_data.empty or self.sentiment_col not in category_data.columns:
            return 0.0, "neutral"  # Default if no data

        # Prepare data for Prophet
        category_data = category_data[[self.time_col, self.sentiment_col]].copy()
        category_data["ds"] = pd.to_datetime(category_data[self.time_col], unit="s")
        category_data["y"] = category_data[self.sentiment_col]

        model = Prophet()
        model.fit(category_data[["ds", "y"]])

        # Predict future sentiment
        future = model.make_future_dataframe(periods=self.forecast_days)
        forecast = model.predict(future)
        forecasted_sentiment = forecast["yhat"].iloc[-1]

        # Classify sentiment
        if forecasted_sentiment > 0.1:
            sentiment_label = "positive"
        elif forecasted_sentiment < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return forecasted_sentiment, sentiment_label
