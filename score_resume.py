import pdfplumber
import joblib

def extract_sections(text):
    sections = {
        "skills": "",
        "education": "",
        "experience": "",
        "projects": "",
    }

    lines = text.lower().split('\n')
    current_section = None

    for line in lines:
        line = line.strip()
        if "skill" in line:
            current_section = "skills"
        elif "education" in line:
            current_section = "education"
        elif "experience" in line:
            current_section = "experience"
        elif "project" in line:
            current_section = "projects"
        elif current_section:
            sections[current_section] += line + " "

    return sections

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_resume_score(pdf_path, model, vectorizer):
    resume_text = extract_text_from_pdf(pdf_path)
    sections = extract_sections(resume_text)
    combined = sections["skills"] + " " + sections["education"] + " " + sections["experience"] + " " + sections["projects"]
    features = vectorizer.transform([combined])
    score = model.predict(features)[0]
    return score

if __name__ == "__main__":
    pdf_path = "testresume.pdf"  # Change this to your test file
    model_path = "models/resume_score_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"

    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
    score = predict_resume_score(pdf_path, model, vectorizer)
    print(f"Predicted ATS Score: {score:.2f}")
