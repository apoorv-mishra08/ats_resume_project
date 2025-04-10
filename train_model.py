import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import joblib
import os
import numpy as np

def train_resume_model(csv_path, model_path, vectorizer_path):
    # Load the dataset
    df = pd.read_csv(csv_path)
    df = df.fillna("")

    # Combine structured fields
    df['combined'] = (
        df['skills'] + " " +
        df['education'] + " " +
        df['experience'] + " " +
        df['projects']
    )

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['combined'])
    y = df['ats_score']

    # Split for training/testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    mean_cv = np.mean(cv_scores)

    print(f"R² Score: {r2:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Cross-Validation R² (5-fold): {mean_cv:.2f}")

    # Save model and vectorizer
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved at {model_path}")
    print(f"Vectorizer saved at {vectorizer_path}")

if __name__ == "__main__":
    csv_path = "structured_resume_dataset.csv"
    model_path = "models/resume_score_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    train_resume_model(csv_path, model_path, vectorizer_path)
