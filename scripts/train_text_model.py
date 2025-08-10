# scripts/train_text_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ✅ Import preprocessing so cleaned_text is created
from scripts.preprocess_text import load_text_data 

# Load with preprocessing
df = load_text_data("preprocessed_data.csv")  # returns DataFrame with 'cleaned_text'
X = df['cleaned_text']
y = df['label']  # <-- make sure this matches your CSV label column name

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "models/text_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("✅ Text model and vectorizer saved in models/")
