from sentiment_analyser.pipeline_builder import SentimentPipelineBuilder


class SentimentPipelineDirector:
    """Director - Ensures the pipeline runs in the correct order."""
    
    def __init__(self, builder: SentimentPipelineBuilder):
        self.builder = builder

    def construct_pipeline(self, text: str):
        """Runs all necessary pipeline steps in order."""
        return (
            self.builder
            .preprocess(text)
            .validate()
            .analyze_sentiment()
            .get_result()
        )
