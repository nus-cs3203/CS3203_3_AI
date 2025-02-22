from sentiment_analyser.classifiers.bert_classifier import BERTSentimentClassifier
from sentiment_analyser.classifiers.vader_classifier import VaderSentimentClassifier
from sentiment_analyser.pipeline_builder import SentimentPipelineBuilder
from sentiment_analyser.pipeline_director import SentimentPipelineDirector


if __name__ == "__main__":
    text = "I absolutely love this! Best experience ever."

    # Use BERT classifier with the pipeline
    bert_builder = SentimentPipelineBuilder(BERTSentimentClassifier())
    bert_director = SentimentPipelineDirector(bert_builder)
    final_bert_result = bert_director.construct_pipeline(text)
    
    print("Final Sentiment (BERT):", final_bert_result)

    # Use VADER classifier with the pipeline
    vader_builder = SentimentPipelineBuilder(VaderSentimentClassifier())
    vader_director = SentimentPipelineDirector(vader_builder)
    final_vader_result = vader_director.construct_pipeline(text)

    print("Final Sentiment (VADER):", final_vader_result)
