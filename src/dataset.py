# src/dataset.py
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from .preprocess import preprocess_text
import torch

class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts.tolist(), truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in item.items()}

    def __len__(self):
        return len(self.labels)

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    # Map class labels to integers
    data['label'] = data['class'].map({0: 0, 1: 1, 2: 2})
    # Preprocess text
    data['clean_text'] = data['tweet'].apply(preprocess_text)
    return data