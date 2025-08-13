# scripts/train_audio_model.py
import os
import librosa
import numpy as np
import joblib
import warnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

AUDIO_DIR = "data/audio data"
MODEL_PATH = "audio_model_boosted.pkl"

# --------------------------
# AUDIO AUGMENTATION HELPERS
# --------------------------
def augment_audio(y, sr):
    """Return list of augmented audio versions."""
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
    augmented.append(y * 1.2)  # volume up
    augmented.append(y * 0.8)  # volume down
    return augmented

# --------------------------
# FEATURE EXTRACTION
# --------------------------
def extract_features(y, sr):
    """Extract MFCC, delta, delta-delta, and Mel features."""
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Aggregate statistics
        feats = np.hstack([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(delta, axis=1), np.std(delta, axis=1),
            np.mean(delta2, axis=1), np.std(delta2, axis=1),
            np.mean(mel_db, axis=1), np.std(mel_db, axis=1)
        ])
        return feats
    except Exception as e:
        print(f"⚠️ Feature extraction failed: {e}")
        return None

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
                # Augmentations
                for aug_audio in augment_audio(audio, sr):
                    feats = extract_features(aug_audio, sr)
                    if feats is not None:
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
    print(f"🎵 Loaded {len(X)} samples ({np.sum(y==1)} dementia, {np.sum(y==0)} control)")

    if len(X) == 0:
        raise ValueError("No audio data loaded! Check paths & file formats.")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class imbalance
    scale_pos_weight = np.sum(y_train==0) / np.sum(y_train==1)

    # XGBoost with Grid Search
    params = {
        "n_estimators": [200],
        "max_depth": [6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    grid = GridSearchCV(
        model,
        param_grid=params,
        scoring="f1_macro",
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"✅ Best Params: {grid.best_params_}")

    # Evaluation
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ Macro F1-score: {macro_f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model pipeline
    joblib.dump({"scaler": scaler, "pca": pca, "model": best_model}, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_audio_model(AUDIO_DIR)
