from insight_generator.base_decorator import InsightDecorator
from prophet import Prophet
import pandas as pd

class TopicSentimentForecastDecorator(InsightDecorator):
    def __init__(self, wrapped_insight_generator, historical_data, log_file="sentiment_forecasts.txt", 
                 time_col="created_utc", sentiment_col="sentiment_title_selftext_polarity", 
                 category_col="Domain Category", forecast_days=7):
        """
        Initialize with historical sentiment data for forecasting.
        :param wrapped_insight_generator: Base insight generator to wrap
        :param historical_data: A DataFrame containing historical sentiment data
        :param log_file: File to log sentiment forecasts
        :param time_col: Column containing timestamps
        :param sentiment_col: Column containing sentiment polarity scores
        :param category_col: Column containing the topic/category
        :param forecast_days: Number of future days to forecast
        """
        super().__init__(wrapped_insight_generator)
        self.historical_data = historical_data
        self.log_file = log_file
        self.time_col = time_col
        self.sentiment_col = sentiment_col
        self.category_col = category_col
        self.forecast_days = forecast_days  # Customizable forecasting window

    def extract_insights(self, post):
        """
        Extract insights including sentiment forecast based on historical data.
        :param post: A single Reddit post (row from DataFrame)
        :return: Insights including sentiment forecast
        """
        insights = super().extract_insights(post)
        
        # Extract topic/category
        topic = post.get(self.category_col, "").lower()
        
        # Forecast sentiment for the topic
        sentiment_score, sentiment_label = self.forecast_sentiment(topic)
        
        # Log the forecasted sentiment
        self.log_forecast(topic, sentiment_score, sentiment_label)
        
        insights["sentiment_forecast"] = {
            "score": sentiment_score,
            "label": sentiment_label
        }
        return insights

    def forecast_sentiment(self, topic):
        """
        Forecast sentiment for the topic using Prophet based on historical sentiment data.
        :param topic: The topic/category of the post (e.g., transport, health)
        :return: Tuple (forecasted sentiment score, forecasted sentiment label)
        """
        # Filter historical data based on the topic/category
        topic_data = self.historical_data[self.historical_data[self.category_col] == topic]

        if topic_data.empty:
            return 0.0, "neutral"  # Default if no historical data is available

        # Ensure required sentiment column exists
        if self.sentiment_col not in topic_data.columns:
            raise ValueError(f"Missing '{self.sentiment_col}' column in historical_data!")

        # Prepare data for Prophet
        topic_data = topic_data[[self.time_col, self.sentiment_col]]
        topic_data["ds"] = pd.to_datetime(topic_data[self.time_col], unit="s")  # Convert to datetime
        topic_data["y"] = topic_data[self.sentiment_col]  # Prophet target variable
        topic_data = topic_data[["ds", "y"]]  # Ensure proper format

        # Train Prophet model
        model = Prophet()
        model.fit(topic_data)

        # Generate future dates
        future = model.make_future_dataframe(periods=self.forecast_days)

        # Forecast sentiment
        forecast = model.predict(future)
        forecasted_sentiment = forecast["yhat"].iloc[-1]

        # Classify the forecasted sentiment
        if forecasted_sentiment > 0.1:
            sentiment_label = "positive"
        elif forecasted_sentiment < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return forecasted_sentiment, sentiment_label

    def log_forecast(self, topic, score, label):
        """
        Logs the forecasted sentiment score and label to a file.
        :param topic: Topic/category of the sentiment forecast
        :param score: Forecasted sentiment score
        :param label: Forecasted sentiment label
        """
        with open(self.log_file, "a") as file:
            file.write(f"Topic: {topic}, Score: {score:.4f}, Label: {label}\n")
