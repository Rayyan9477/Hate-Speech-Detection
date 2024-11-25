# app.py
import streamlit as st

st.set_page_config(page_title="Hate Speech Detection", page_icon="📝", layout="wide")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

from src.preprocess import preprocess_text
from rejoin import reassemble_files

@st.cache_resource
def load_or_train_model():
    model_dirs = ['./fine_tuned_model', './results/checkpoint-4957']
    for directory in model_dirs:
        reassemble_files(directory)

    tokenizer = AutoTokenizer.from_pretrained('./fine_tuned_model')
    model = AutoModelForSequenceClassification.from_pretrained(
        './fine_tuned_model',
        ignore_mismatched_sizes=True
    )
    return tokenizer, model

# Load the model
tokenizer, model = load_or_train_model()
model.eval()
labels = {0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"}

# Initialize or get the background color from session state
if 'bg_color' not in st.session_state:
    st.session_state['bg_color'] = '#ffffff'  # Default white background
bg_color = st.session_state['bg_color']

# Custom CSS for styling
st.markdown(f"""
    <style>
    /* Set background color */
    body {{
        background-color: {bg_color};
    }}
    /* Center the main content */
    .main {{
        max-width: 800px;
        margin: 0 auto;
    }}
    /* Style the predict button */
    .stButton>button {{
        color: white !important;
        background-color: #4CAF50;
        padding: 0.5em 1em;
        border-radius: 5px;
        border: none;
    }}
    /* Style the text area */
    textarea {{
        font-size: 1em;
    }}
    /* Footer style */
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: black;
        text-align: center;
        padding: 0.5em 0;
        font-size: 0.8em;
    }}
    /* Header style */
    .header {{
        text-align: center;
        padding: 1em 0;
    }}
    </style>
    """, unsafe_allow_html=True)

# Main content
st.title("📝 Hate Speech Detection")

st.markdown("""
<div class="header">
    <h3>Welcome to the Hate Speech Detection App</h3>
    <p>This tool uses a fine-tuned Transformer model to classify text into:</p>
    <ul>
        <li><strong>Hate Speech</strong></li>
        <li><strong>Offensive Language</strong></li>
        <li><strong>No Hate and Offensive</strong></li>
    </ul>
    <p>Feel free to input any text and see the model's prediction.</p>
</div>
""", unsafe_allow_html=True)

# Input text from user
st.header("Enter Text to Analyze")
text = st.text_area("Type or paste text here:", height=200)

# Add a button and display a spinner while processing
if st.button("Detect Hate Speech"):
    if text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing...'):
            # Preprocess the input text
            clean_text = preprocess_text(text)
            # Tokenize input
            inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[0][predicted_class].item()

        # Update background color based on prediction
        if predicted_class in [0, 1]:
            # Offensive language detected, set background to red
            st.session_state['bg_color'] = '#ffcccc'  # Light red
        else:
            # No offensive language, set background to green
            st.session_state['bg_color'] = '#ccffcc'  # Light green

        # Display the results
        st.success(f"**Prediction:** {labels[predicted_class]}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        # Display probabilities in a bar chart
        st.subheader("Classification Probabilities")
        probabilities = probabilities.squeeze().tolist()
        st.bar_chart({
            'Labels': list(labels.values()),
            'Probability': probabilities
        })

# Footer with contact information
st.markdown("""
<div class="footer">
    <p>Developed by <a href="https://www.linkedin.com/in/rayyan-ahmed9477/" target="_blank">Rayyan Ahmed</a> |
    <a href="mailto:rayyanahmed265@yahoo.com">rayyanahmed265@yahoo.com</a> |
    <a href="https://github.com/Rayyan9477" target="_blank">GitHub</a></p>
</div>
""", unsafe_allow_html=True)