# scripts/train_audio_model.py
import os
import librosa
import numpy as np
import pandas as pd
import joblib
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_DIR = "data/audio data"
MODEL_PATH = "audio_model.pkl"

# --------------------------
# AUDIO AUGMENTATION HELPERS
# --------------------------
def augment_audio(y, sr):
    augmented = [y]  # original
    try:
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=2))
        augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))
    except:
        pass
    try:
        augmented.append(librosa.effects.time_stretch(y, rate=0.9))
        augmented.append(librosa.effects.time_stretch(y, rate=1.1))
    except:
        pass
    noise = np.random.normal(0, 0.005, y.shape)
    augmented.append(y + noise)
    return augmented

# --------------------------
# FEATURE EXTRACTION
# --------------------------
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    feats = np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1),
        np.mean(mel_db, axis=1), np.std(mel_db, axis=1)
    ])
    return feats

def feature_names():
    cols = []
    for prefix, dim in [("mfcc", 20), ("delta", 20), ("delta2", 20), ("mel", 40)]:
        cols += [f"{prefix}_mean_{i}" for i in range(dim)]
        cols += [f"{prefix}_std_{i}" for i in range(dim)]
    return cols

# --------------------------
# LOAD AUDIO DATA
# --------------------------
def load_audio_dataset(audio_dir):
    X, y = [], []
    for label_name, label_val in [("dementia", 1), ("control", 0)]:
        folder_path = os.path.join(audio_dir, label_name)
        if not os.path.exists(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp3")]
        print(f"Loading {label_name}: {len(files)} files")

        for f in tqdm(files, desc=f"Loading {label_name}"):
            file_path = os.path.join(folder_path, f)
            try:
                audio, sr = librosa.load(file_path, sr=None)
                for aug_audio in augment_audio(audio, sr):
                    feats = extract_features(aug_audio, sr)
                    X.append(feats)
                    y.append(label_val)
            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")

    return np.array(X), np.array(y)

# --------------------------
# TRAINING PIPELINE
# --------------------------
def train_audio_model(audio_dir):
    print(f"🎯 Loading audio data from: {audio_dir}")
    X, y = load_audio_dataset(audio_dir)
    print(f"🎵 Loaded {len(X)} samples")

    if len(X) == 0:
        raise ValueError("No audio data loaded!")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA optional
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    # Balance
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ Macro F1: {f1:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save with schema
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "feature_names": feature_names()
    }, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_audio_model(AUDIO_DIR)
