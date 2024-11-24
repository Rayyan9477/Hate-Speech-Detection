# src/train.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from .dataset import HateSpeechDataset, load_dataset
import os

def train_model():
    # Load and preprocess dataset
    data = load_dataset('data/dataset.csv')
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("KoalaAI/HateSpeechDetector")
    model = AutoModelForSequenceClassification.from_pretrained("KoalaAI/HateSpeechDetector", num_labels=3)

    # Create datasets
    train_dataset = HateSpeechDataset(
        train_data['clean_text'], train_data['label'], tokenizer
    )
    val_dataset = HateSpeechDataset(
        val_data['clean_text'], val_data['label'], tokenizer
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        save_total_limit=1,
    )

    # Define compute_metrics function
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        accuracy = (preds == p.label_ids).astype(float).mean().item()
        return {'accuracy': accuracy}

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')