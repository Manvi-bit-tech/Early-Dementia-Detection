import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ✅ Import from scripts/
from scripts.preprocess_audio import load_audio_dataset

# Load audio data
X_audio, y_audio = load_audio_dataset("data/audio data")  # Adjust path if needed

# Flatten MFCCs for classical ML models
n_samples = X_audio.shape[0]
X_audio_flat = X_audio.reshape(n_samples, -1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_audio_flat, y_audio, test_size=0.2, stratify=y_audio, random_state=42
)

# Train model
clf = SVC(kernel="linear", probability=True)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/audio_model.pkl")
print("✅ Audio model saved.")

