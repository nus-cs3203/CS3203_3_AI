from insight_generator.insight_interface import InsightGenerator

class InsightDecorator(InsightGenerator):
    def __init__(self, wrapped: InsightGenerator):
        """
        A decorator that wraps the base insight generator to allow for extended functionalities.
        :param wrapped: An instance of InsightGenerator that is being wrapped.
        """
        self._wrapped = wrapped  # The wrapped object could be a concrete component or another decorator

    def extract_insights(self, post):
        return self._wrapped.extract_insights(post)  # Delegates to the wrapped object
