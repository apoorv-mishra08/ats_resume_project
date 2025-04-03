import os
import pdfplumber
import joblib

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def load_model_and_vectorizer(model_path, vectorizer_path):
    """
    Loads the saved ML model and TF-IDF vectorizer.
    """
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer file not found.")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_resume_score(resume_text, model, vectorizer):
    """
    Predicts the ATS score of a given resume.
    """
    features = vectorizer.transform([resume_text])
    score = model.predict(features)[0]
    return round(score, 2)


if __name__ == "__main__":
    # Step 1: Define file name here
    test_resume_path = "Siddarth Ambastha.pdf"  

    # Step 2: Load the model and vectorizer
    model_path = "models/resume_score_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    model, vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)

    # Step 3: Extract 
    try:
        resume_text = extract_text_from_pdf(test_resume_path)
        if resume_text:
            # Step 4: Predict ATS score
            score = predict_resume_score(resume_text, model, vectorizer)
            print("\n Predicted ATS Score:", score, "/ 100")
        else:
            print(" Resume text could not be extracted.")
    except Exception as e:
        print(" Error:", str(e))
