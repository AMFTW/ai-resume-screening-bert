import streamlit as st
import torch
import numpy as np
import joblib
from transformers import BertTokenizer, BertForSequenceClassification

# Page config
st.set_page_config(page_title="AI Resume Screening", layout="centered")

st.title("AI Resume Screening System")
st.write("Paste a resume below to predict job category")

# Load model + tokenizer
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("model/")
    tokenizer = BertTokenizer.from_pretrained("model/")
    return model, tokenizer

model, tokenizer = load_model()

# Load label encoder
@st.cache_resource
def load_labels():
    return joblib.load("model/label_encoder.pkl")

label_encoder = load_labels()
labels = label_encoder.classes_

# Input
resume_text = st.text_area("Resume Text", height=200)

if st.button("Predict"):

    if resume_text.strip() == "":
        st.warning("Please paste a resume")
    else:
        # Tokenize input
        inputs = tokenizer(
            resume_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # Model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)

        probs = probs[0].cpu().numpy()

        # Get top prediction
        pred_index = np.argmax(probs)
        pred_label = labels[pred_index]
        confidence = probs[pred_index]

        # Display main prediction
        st.success(f"Predicted Category: {pred_label}")
        st.write(f"Confidence: {confidence:.2f}")

        # Top 3 predictions (FIXED)
        st.subheader("Top Predictions")

        top_indices = probs.argsort()[-3:][::-1]

        for idx in top_indices:
            st.write(f"{labels[idx]}: {probs[idx]:.2f}")