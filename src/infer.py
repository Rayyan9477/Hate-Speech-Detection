# src/infer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .preprocess import preprocess_text

def classify_text(text):
    tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
    model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model')
    model.eval()

    labels = model.config.id2label

    # Preprocess the input text
    text = preprocess_text(text)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()

    return labels[str(predicted_class)], confidence

if __name__ == '__main__':
    text = input("Enter text to classify: ")
    label, confidence = classify_text(text)
    print(f"Prediction: {label} (Confidence: {confidence*100:.2f}%)")