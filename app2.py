# app2.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocess import preprocess_text
from src.train import train_model
import os

# Function to load or fine-tune the model
@st.cache_resource
def load_or_train_model():
    model_dir = './fine_tuned_model'
    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            ignore_mismatched_sizes=True
        )
    else:
        with st.spinner('Fine-tuning the model, please wait...'):
            train_model()
            tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
            model = AutoModelForSequenceClassification.from_pretrained(
                './fine_tuned_model',
                ignore_mismatched_sizes=True
            )
    return tokenizer, model

# Load or fine-tune the model
tokenizer, model = load_or_train_model()
model.eval()
labels = {str(i): label for i, label in enumerate(["Hate Speech", "Offensive Language", "No Hate and Offensive"])}

# Streamlit App
st.title("Hate Speech Detection")

# Input text from user
text = st.text_area("Enter the text to analyze:")

if st.button("Detect Hate Speech"):
    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess the input text
        clean_text = preprocess_text(text)
        # Tokenize input
        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[0][predicted_class].item()
        st.write(f"**Prediction:** {labels[str(predicted_class)]}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")