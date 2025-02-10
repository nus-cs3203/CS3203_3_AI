from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json

def predict_zero_shot(texts):
    # Initialize zero-shot classifier
    model_name = "facebook/bart-large-mnli"
    classifier = pipeline("zero-shot-classification", model=model_name)
    
    predictions = []
    for text in texts:
        result = classifier(
            text,
            candidate_labels=["Direct Complaint", "Not Complaint"],
            hypothesis_template="This post expresses {}",
        )
        # Predict 1 if "Direct Complaint" has highest score
        pred = 1 if result['labels'][0] == "Direct Complaint" else 0
        predictions.append(pred)
    
    return predictions

def compute_metrics(predictions, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # Load all 200 rows
    df = pd.read_csv('200rows_Filtered_Data.csv')
    df['is_complaint'] = df['complaint_categories'].apply(lambda x: 1 if x == 'Direct Complaint' else 0)
    df['text'] = df['title'] + ' ' + df['selftext'].fillna('')
    
    print("\nEvaluating Zero-shot BART-large-MNLI...")
    zero_shot_preds = predict_zero_shot(df['text'].tolist())
    zero_shot_metrics = compute_metrics(zero_shot_preds, df['is_complaint'].tolist())
    
    # Print results
    print("\nZero-shot BART-large-MNLI metrics (all data):")
    print(json.dumps(zero_shot_metrics, indent=2))
    
    print("\nPreviously obtained DistilBERT metrics:")
    print("Test set results:")
    print(json.dumps({
        'accuracy': 0.7333,
        'f1': 0.3333,
        'precision': 1.0000,
        'recall': 0.2000
    }, indent=2))
    
    print("\nTraining set results:")
    print(json.dumps({
        'accuracy': 0.8571,
        'f1': 0.5652,
        'precision': 0.8125,
        'recall': 0.4333
    }, indent=2))

if __name__ == "__main__":
    main() 