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

# Job role and description input
st.subheader("🧾 Job Role & Description")
job_role = st.text_input("Enter the Job Role (e.g., Data Analyst)")
job_description = st.text_area("Paste the Job Description below")




if uploaded_file is not None and job_description.strip():
    try:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.subheader(" Extracted Resume Preview:")
        st.text_area("Resume Text", resume_text[:1000], height=250)

        model, vectorizer = load_model()
        score = predict_score(resume_text, model, vectorizer)

        st.subheader("ATS Score:")
        st.metric(label="Predicted ATS Score", value=f"{score} / 100")

        # Job match using cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        job_features = vectorizer.transform([job_description])
        resume_features = vectorizer.transform([resume_text])

        match_score = cosine_similarity(resume_features, job_features)[0][0] * 100
        match_score = round(match_score, 2)

        st.subheader(" JD Match Score:")
        st.metric(label=f"Match with {job_role}", value=f"{match_score}%")

    except Exception as e:
        st.error(f" Error processing file: {e}")
else:
    st.info("Please upload a resume and enter a job description.")
