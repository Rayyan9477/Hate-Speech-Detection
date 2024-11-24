# src/preprocess.py
import re
import emoji

def preprocess_text(text):
    # Remove emojis
    text = emoji.replace_emoji(text, replace='')
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text.strip()