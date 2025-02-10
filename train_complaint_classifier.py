from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
import numpy as np

# Configure device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Create custom dataset class
class ComplaintDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def print_gpu_utilization():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

def prepare_data():
    # Load and preprocess data
    df = pd.read_csv('200rows_Filtered_Data.csv')
    
    # Convert to binary classification (complaint vs non-complaint)
    df['is_complaint'] = df['complaint_categories'].apply(lambda x: 1 if x == 'Direct Complaint' else 0)
    
    # Combine title and selftext for the input text
    df['text'] = df['title'] + ' ' + df['selftext'].fillna('')
    
    # Split into train (70%), validation (15%), and test (15%) sets
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['text'].tolist(),
        df['is_complaint'].tolist(),
        test_size=0.3,
        random_state=42
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=42
    )
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Add weighted loss computation
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Add class weights
        weights = torch.tensor([1.0, 3.0], device=device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def main():
    print("Starting fine-tuning process...")
    print_gpu_utilization()
    
    # Prepare data
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = prepare_data()
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"  # Much smaller than BART-large
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        dropout=0.3  # DistilBERT uses a single dropout parameter
    )
    model = model.to(device)
    
    # Create datasets
    train_dataset = ComplaintDataset(train_texts, train_labels, tokenizer)
    val_dataset = ComplaintDataset(val_texts, val_labels, tokenizer)
    test_dataset = ComplaintDataset(test_texts, test_labels, tokenizer)
    
    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_dataset)} examples")
    print(f"Validation: {len(val_dataset)} examples")
    print(f"Test: {len(test_dataset)} examples")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=50,
        learning_rate=2e-5,
        weight_decay=0.2,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        gradient_accumulation_steps=1,
        lr_scheduler_type="linear",
        warmup_ratio=0.1
    )
    
    # Initialize trainer with early stopping
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("Test set results:", test_results)
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_results = trainer.evaluate(train_dataset)
    print("Training set results:", train_results)
    
    print("\nFine-tuning complete!")

if __name__ == "__main__":
    main() 