import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_resume_model(csv_path, model_path, vectorizer_path):
    
    df = pd.read_csv(csv_path)
    texts = df['resume_text'].fillna("")
    scores = df['ats_score']

    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    y = scores

    # Split data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Model R2 Score: {r2_score(y_test, y_pred):.2f}")
    print(f"Model RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")

    # Save model and vectorizer
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved at {model_path}")
    print(f"Vectorizer saved at {vectorizer_path}")

if __name__ == "__main__":
    csv_path = "princeton_resume_dataset.csv"
    model_path = "models/resume_score_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    train_resume_model(csv_path, model_path, vectorizer_path)
