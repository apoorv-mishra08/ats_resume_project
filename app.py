import streamlit as st
import pdfplumber
import joblib
import os

@st.cache_resource
def load_model():
    model = joblib.load("models/resume_score_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def predict_score(text, model, vectorizer):
    features = vectorizer.transform([text])
    score = model.predict(features)[0]
    return round(score, 2)


st.set_page_config(page_title="AI Resume Scorer", layout="centered")
st.title(" ATS Resume Score Predictor")
st.markdown("Upload your resume PDF and get a predicted ATS score based on top-tier standards like Princeton.")

uploaded_file = st.file_uploader("Upload Resume (PDF only)", type="pdf")

if uploaded_file is not None:
    try:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.subheader(" Extracted Resume Preview:")
        st.text_area("Text", resume_text[:1000], height=250)

        model, vectorizer = load_model()
        score = predict_score(resume_text, model, vectorizer)

        st.subheader("📊 ATS Score:")
        st.metric(label="Predicted ATS Score", value=f"{score} / 100")

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
