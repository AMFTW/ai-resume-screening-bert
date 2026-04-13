import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# page config
st.set_page_config(page_title="AI Resume Screening", layout="centered")

st.title("AI Resume Screening System")
st.write("Paste a resume below to predict job category")

# load model from HuggingFace (no local files)
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# dummy labels (replace later if you retrain model)
labels = [
    "ADVOCATE",
    "ENGINEERING",
    "DATA SCIENCE",
    "BUSINESS DEVELOPMENT",
    "HR",
    "SALES",
    "FINANCE",
    "TEACHING",
    "HEALTHCARE",
    "DESIGN"
]

# input
resume_text = st.text_area("Resume Text")

if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text")
    else:
        # tokenize
        inputs = tokenizer(
            resume_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # model inference
        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=1)
        pred_index = torch.argmax(probs).item()
        confidence = probs[0][pred_index].item()

        # result
        st.success(f"Predicted Category: {labels[pred_index]}")
        st.write(f"Confidence: {confidence:.2f}")

        # top 3 predictions
        st.subheader("Top Predictions")
        top3 = torch.topk(probs, 3)

        for i in range(3):
            idx = top3.indices[0][i].item()
            score = top3.values[0][i].item()
            st.write(f"{labels[idx]}: {score:.2f}")