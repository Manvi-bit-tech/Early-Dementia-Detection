# scripts/train_fusion_model.py

import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from scripts.preprocess_text import load_text_data
from scripts.preprocess_audio import load_audio_dataset


def normalize_id(id_val):
    """Convert IDs to a comparable format (string, lowercase, no extension)."""
    return str(id_val).strip().lower().replace(".wav", "").replace(".txt", "")


# 1️⃣ Load text data
df_text = load_text_data("preprocessed_data_cleaned.csv")
df_text["id"] = df_text["id"].apply(normalize_id)
df_text["label"] = df_text["label"].astype(int)

# 2️⃣ Load audio data
X_audio, y_audio, audio_ids = load_audio_dataset("data/audio data/", return_ids=True)
n_samples, n_mfcc, n_frames = X_audio.shape
X_audio_flat = X_audio.reshape(n_samples, n_mfcc * n_frames)

df_audio = pd.DataFrame(X_audio_flat)
df_audio["id"] = pd.Series(audio_ids).apply(normalize_id)
df_audio["label"] = pd.Series(y_audio).astype(int)

# 3️⃣ Debug: Check ID overlaps before merging
text_ids = set(df_text["id"])
audio_ids_set = set(df_audio["id"])
common_ids = text_ids.intersection(audio_ids_set)

print(f"📝 Text dataset: {len(text_ids)} unique IDs")
print(f"🎵 Audio dataset: {len(audio_ids_set)} unique IDs")
print(f"🔗 Common IDs: {len(common_ids)}")
if len(common_ids) == 0:
    print("⚠️ No matching IDs found between text & audio datasets! Check filenames/IDs.")
    # Optionally exit early
    # exit()

# 4️⃣ Merge on ID only, then verify labels
df_merged = pd.merge(df_text, df_audio, on="id", suffixes=("_text", "_audio"))

# Keep only rows where labels match
df_merged = df_merged[df_merged["label_text"] == df_merged["label_audio"]].copy()
df_merged.rename(columns={"label_text": "label"}, inplace=True)
df_merged.drop(columns=["label_audio"], inplace=True)

print(f"✅ Merged dataset after label match: {df_merged.shape[0]} samples")

if df_merged.empty:
    raise ValueError("❌ Merged dataset is empty after label matching!")

# 5️⃣ Prepare features
vectorizer = TfidfVectorizer(max_features=5000)
X_text_tfidf = vectorizer.fit_transform(df_merged["cleaned_text"])

audio_feature_cols = [col for col in df_audio.columns if str(col).isdigit()]
X_audio_numeric = df_merged[audio_feature_cols].to_numpy(dtype=np.float32)

X_combined = hstack([csr_matrix(X_text_tfidf, dtype=np.float32),
                     csr_matrix(X_audio_numeric, dtype=np.float32)], format="csr")
y_combined = df_merged["label"].to_numpy()

# 6️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

# 7️⃣ Train model
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# 8️⃣ Evaluation
y_pred = clf.predict(X_test)
print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))
