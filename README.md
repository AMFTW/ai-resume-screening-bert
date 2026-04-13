# AI Resume Screening System (BERT)

An end-to-end Natural Language Processing (NLP) project that classifies resumes into job categories using a fine-tuned BERT model. The system provides predictions along with confidence scores and top alternative categories.

---

## Live Demo

(Add your deployed Streamlit link here)

---

## Problem Statement

Recruiters often spend significant time manually screening resumes. This project automates the process by predicting the most relevant job category for a given resume using a deep learning model.

---

## Approach

* Converted raw resume text files into a structured dataset
* Cleaned text by removing emails, phone numbers, and special characters
* Encoded job categories into numerical labels
* Fine-tuned a pre-trained BERT model for text classification
* Built an interactive Streamlit app for real-time predictions

---

## Tech Stack

* Python
* PyTorch
* HuggingFace Transformers (BERT)
* Scikit-learn
* Pandas
* Streamlit

---

## Project Structure

```
resume-bert-project/
│
├── app.py
├── train.py
├── clean_data.py
├── prepare_data.py
├── requirements.txt
│
├── data/
│   ├── cleaned_resume.csv
│
├── model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer files
│   ├── label_encoder.pkl
│
└── README.md
```

---

## Features

* Predict job category from resume text
* Displays model confidence score
* Shows top 3 predicted categories
* Clean and interactive UI
* End-to-end ML pipeline (data → training → deployment)

---

## Example Output

* **Predicted Category:** Business Development
* **Confidence:** 0.05

**Top Predictions:**

* Business Development: 0.05
* Engineering: 0.05
* Chef: 0.05

---

## Limitations

* Model performance depends on dataset quality
* Some categories may be misclassified due to class imbalance
* Low confidence scores indicate uncertainty in prediction

---

## How to Run Locally

```bash
# clone repo
git clone https://github.com/YOUR_USERNAME/resume-bert-project.git

# go into folder
cd resume-bert-project

# install dependencies
pip install -r requirements.txt

# run app
python -m streamlit run app.py
```

---

## Future Improvements

* Improve dataset quality and balance
* Increase training epochs for better accuracy
* Add resume-job matching system
* Deploy using scalable cloud infrastructure

---

## Author

Aryaman Melwani
Developed as part of an AI/ML portfolio project to demonstrate NLP, deep learning, and deployment skills.

---
