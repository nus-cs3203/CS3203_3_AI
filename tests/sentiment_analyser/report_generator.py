import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sentence_transformers import SentenceTransformer, util

# Load datasets
df_input = pd.read_csv("tests/sentiment_analyser/data/2022_2025_merged_llm_pre_500.txt")
df_llm_polarity = pd.read_csv("tests/sentiment_analyser/data/sentiment_llm_annotated_1.csv")
df_llm_emotion = pd.read_csv("tests/sentiment_analyser/data/sentiment_llm_annotated_2.csv")
df_bert_results = pd.read_csv("tests/sentiment_analyser/data/2022_2025_merged_llm_pre_500_bert.txt")
df_vader_results = pd.read_csv("tests/sentiment_analyser/data/2022_2025_merged_llm_pre_500_vader.txt")
df_roberta_results = pd.read_csv("tests/sentiment_analyser/data/2022_2025_merged_llm_pre_500_roberta.txt")
df_distilroberta_results = pd.read_csv("tests/sentiment_analyser/data/2022_2025_merged_llm_pre_500_distilroberta.txt")


# Prepare data for comparison
df_bert_results = df_bert_results.rename(columns={"title_with_desc_label": "label", "title_with_desc_score": "score"})
df_vader_results = df_vader_results.rename(columns={"title_with_desc_label": "label", "title_with_desc_score": "score"})
df_roberta_results = df_roberta_results.rename(columns={"title_with_desc_emotion": "label", "title_with_desc_score": "score"})
df_distilroberta_results = df_distilroberta_results.rename(columns={"title_with_desc_emotion": "label", "title_with_desc_score": "score"})
df_llm_polarity = df_llm_polarity.rename(columns={"sentiment_score": "score"})
df_llm_emotion = df_llm_emotion.rename(columns={"confidence_score": "score"})
df_llm_emotion = df_llm_emotion.rename(columns={"emotion": "label"})
df_llm_polarity['label'] = df_llm_polarity['score'].apply(lambda x: 'neutral' if -0.05 <= x <= 0.05 else ('positive' if x > 0 else 'negative'))


# Define a function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    return accuracy, f1, recall, precision

# Calculate metrics for LLM Polarity with BERT and VADER
metrics_llm_bert = calculate_metrics(df_llm_polarity['label'], df_bert_results['label'])
metrics_llm_vader = calculate_metrics(df_llm_polarity['label'], df_vader_results['label'])

# Define a function to calculate the mean absolute error between scores
def calculate_score_difference(df1, df2, score_column='score'):
    return (df1[score_column] - df2[score_column]).abs().mean()

# Calculate score differences for LLM Polarity with BERT and VADER
score_diff_llm_bert = calculate_score_difference(df_llm_polarity, df_bert_results)
score_diff_llm_vader = calculate_score_difference(df_llm_polarity, df_vader_results)

# Print the score differences
print("Score difference for LLM Polarity with BERT: {:.4f}".format(score_diff_llm_bert))
print("Score difference for LLM Polarity with VADER: {:.4f}".format(score_diff_llm_vader))

# Calculate metrics for LLM Emotion with RoBERTa and DistilRoBERTa
metrics_llm_roberta = calculate_metrics(df_llm_emotion['label'], df_roberta_results['label'])
metrics_llm_distilroberta = calculate_metrics(df_llm_emotion['label'], df_distilroberta_results['label'])

# Print the results
print("LLM Polarity with BERT: Accuracy: {:.2f}, F1: {:.2f}, Recall: {:.2f}, Precision: {:.2f}".format(*metrics_llm_bert))
print("LLM Polarity with VADER: Accuracy: {:.2f}, F1: {:.2f}, Recall: {:.2f}, Precision: {:.2f}".format(*metrics_llm_vader))
print("LLM Emotion with RoBERTa: Accuracy: {:.2f}, F1: {:.2f}, Recall: {:.2f}, Precision: {:.2f}".format(*metrics_llm_roberta))
print("LLM Emotion with DistilRoBERTa: Accuracy: {:.2f}, F1: {:.2f}, Recall: {:.2f}, Precision: {:.2f}".format(*metrics_llm_distilroberta))

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define a function to calculate semantic similarity
def calculate_semantic_similarity(df1, df2, text_column='label'):
    embeddings1 = model.encode(df1[text_column].tolist(), convert_to_tensor=True)
    embeddings2 = model.encode(df2[text_column].tolist(), convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_scores.mean().item()

# Calculate semantic similarity for LLM Emotion with RoBERTa and DistilRoBERTa
semantic_similarity_llm_roberta = calculate_semantic_similarity(df_llm_emotion, df_roberta_results)
semantic_similarity_llm_distilroberta = calculate_semantic_similarity(df_llm_emotion, df_distilroberta_results)

# Print the semantic similarity scores
print("Semantic similarity for LLM Emotion with RoBERTa: {:.4f}".format(semantic_similarity_llm_roberta))
print("Semantic similarity for LLM Emotion with DistilRoBERTa: {:.4f}".format(semantic_similarity_llm_distilroberta))