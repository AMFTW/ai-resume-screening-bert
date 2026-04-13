import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

st.set_page_config(page_title="Resume Classifier", layout="centered")

# cache model 
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained("model/")
    tokenizer = BertTokenizer.from_pretrained("model/")
    return model, tokenizer

model, tokenizer = load_model()

# load label encoder
le = joblib.load("model/label_encoder.pkl")

# UI
st.title("AI Resume Screening System")
st.write("Paste a resume below to predict job category")

# input
text = st.text_area("Resume Text")

# prediction
if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter resume text")

    else:
        # tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        # model output
        outputs = model(**inputs)

        # main prediction
        prediction = torch.argmax(outputs.logits, dim=1).item()
        category = le.inverse_transform([prediction])[0]

        # probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        confidence = probs[prediction].item()

        st.success(f"Predicted Category: {category}")
        st.write(f"Confidence: {confidence:.2f}")

        # top 3 predictions
        st.subheader("Top Predictions")

        top_k = torch.topk(probs, 3)

        for i in range(3):
            label = le.inverse_transform([top_k.indices[i].item()])[0]
            score = top_k.values[i].item()
            st.write(f"{label}: {score:.2f}")