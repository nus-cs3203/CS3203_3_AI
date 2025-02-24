from insight_generator.base_decorator import InsightDecorator
from prophet import Prophet
import pandas as pd

class TopicSentimentForecastDecorator(InsightDecorator):
    def __init__(self, wrapped_insight_generator, historical_data):
        """
        Initialize with historical sentiment data for forecasting.
        :param wrapped_insight_generator: The base insight generator to wrap
        :param historical_data: A DataFrame containing historical sentiment data for forecasting
        """
        super().__init__(wrapped_insight_generator)
        self.historical_data = historical_data

    def extract_insights(self, post):
        """
        Extract insights including sentiment forecast based on the historical data for the topic.
        :param post: A single post (or row from DataFrame)
        :return: Insights including sentiment forecast
        """
        insights = super().extract_insights(post)
        
        # Analyze sentiment and forecast based on the topic/category
        topic = post.get("Domain Category", "").lower()  # Column for category/topic
        sentiment = post.get("sentiment_title_selftext_label", "").lower()  # Column for sentiment label

        # Forecast sentiment using Prophet model
        sentiment_forecast = self.forecast_sentiment(topic)
        
        insights["sentiment_forecast"] = sentiment_forecast
        return insights

    def forecast_sentiment(self, topic):
        """
        Forecast sentiment for the topic using Prophet based on historical sentiment data.
        :param topic: The topic/category of the post (e.g., transport, health)
        :return: Forecasted sentiment for the future (positive/neutral/negative)
        """
        # Filter historical data based on the topic
        topic_data = self.historical_data[self.historical_data['Domain Category'] == topic]  # Updated to match column name

        # Check if there is any data for the topic
        if topic_data.empty:
            return "neutral"  # If no historical data is available, return neutral sentiment

        # Prepare data for Prophet: Date and sentiment (adjust to use 'sentiment_score' column)
        topic_data = topic_data[['created_utc', 'sentiment_score']]  # Use 'sentiment_score' for forecasting
        topic_data['ds'] = pd.to_datetime(topic_data['created_utc'], unit='s')  # Convert 'created_utc' to datetime
        topic_data['y'] = topic_data['sentiment_score']  # Prophet expects 'y' as the target variable
        topic_data = topic_data[['ds', 'y']]  # Prophet expects 'ds' for date and 'y' for the target value (sentiment score)

        # Train Prophet model
        model = Prophet()
        model.fit(topic_data)

        # Create future dates for prediction (adjust the number of periods as needed)
        future = model.make_future_dataframe(topic_data, periods=5)  # Predict next 5 days

        # Forecast the future sentiment
        forecast = model.predict(future)
        forecasted_sentiment = forecast['yhat'][-1]  # Last predicted sentiment value

        # Classify the forecasted sentiment
        if forecasted_sentiment > 0.1:
            return "positive"
        elif forecasted_sentiment < -0.1:
            return "negative"
        else:
            return "neutral"
