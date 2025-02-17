from insight_generator.insight_interface import InsightGenerator

class InsightDecorator(InsightGenerator):
    def __init__(self, wrapped: InsightGenerator):
        self._wrapped = wrapped

    def extract_insights(self, post):
        return self._wrapped.extract_insights(post)
