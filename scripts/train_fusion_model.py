# scripts/train_fusion_model.py
import os
import joblib
import numpy as np
import pandas as pd
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TEXT_MODEL_PATH = "text_model.pkl"
AUDIO_MODEL_PATH = "audio_model.pkl"
TEXT_FEATURES_PATH = "data/features/text_features.csv"
AUDIO_FEATURES_PATH = "data/features/audio_features.csv"

# ------------------------
# Load models safely
# ------------------------
text_bundle = joblib.load(TEXT_MODEL_PATH)
audio_bundle = joblib.load(AUDIO_MODEL_PATH)

text_model = text_bundle["model"]
vectorizer = text_bundle.get("vectorizer", None)

scaler = audio_bundle.get("scaler", None)
audio_pca = audio_bundle.get("pca", None)
audio_model = audio_bundle["model"]

# ------------------------
# Evaluation helpers
# ------------------------
def evaluate_text():
    df = pd.read_csv(TEXT_FEATURES_PATH)
    y_true = df["label"].values
    X = df.drop(columns=["id", "label"]).values
    y_pred = text_model.predict(X)

    print("\n=== Text Model Evaluation ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

def evaluate_audio():
    df = pd.read_csv(AUDIO_FEATURES_PATH)
    y_true = df["label"].values
    X = df.drop(columns=["id", "label"]).values

    if scaler is not None:
        X = scaler.transform(X)
    if audio_pca is not None:
        X = audio_pca.transform(X)

    y_pred = audio_model.predict(X)

    print("\n=== Audio Model Evaluation ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

# ------------------------
# Fusion inference
# ------------------------
def fused_inference(text_row, audio_row, text_weight=0.6, audio_weight=0.4):
    """Fuse predictions from text and audio."""
    text_prob = text_model.predict_proba([text_row])[0][1]

    audio_input = audio_row.reshape(1, -1)
    if scaler is not None:
        audio_input = scaler.transform(audio_input)
    if audio_pca is not None:
        audio_input = audio_pca.transform(audio_input)
    audio_prob = audio_model.predict_proba(audio_input)[0][1]

    fused_score = text_weight * text_prob + audio_weight * audio_prob
    fused_label = 1 if fused_score >= 0.5 else 0
    return fused_label, text_prob, audio_prob, fused_score

# ------------------------
# Main evaluation + demo
# ------------------------
def evaluate_models():
    evaluate_text()
    evaluate_audio()

    # Demo fusion on common IDs
    print("\n=== Fusion Demo on Common IDs ===")
    df_text = pd.read_csv(TEXT_FEATURES_PATH)
    df_audio = pd.read_csv(AUDIO_FEATURES_PATH)

    common_ids = set(df_text["id"]).intersection(set(df_audio["id"]))
    if not common_ids:
        print("⚠️ No common IDs found between text and audio datasets.")
        return

    sample_ids = random.sample(list(common_ids), min(5, len(common_ids)))
    for sid in sample_ids:
        t_row = df_text[df_text["id"] == sid].drop(columns=["id", "label"]).values[0]
        a_row = df_audio[df_audio["id"] == sid].drop(columns=["id", "label"]).values[0]
        fused_label, text_prob, audio_prob, fused_score = fused_inference(t_row, a_row)

        print(f"\nID: {sid}")
        print(f"  Text prob (dementia): {text_prob:.3f}")
        print(f"  Audio prob (dementia): {audio_prob:.3f}")
        print(f"  → Fused decision: {fused_label} (score={fused_score:.3f})")

if __name__ == "__main__":
    evaluate_models()
    print("\n✅ Fusion system ready for single-subject inference via fused_inference()")
