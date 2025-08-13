# scripts/train_text_model.py
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

# Ensure NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# -----------------------
# Feature extraction helpers
# -----------------------
def extract_pos_features(text):
    """
    Extract POS tag proportions from text.
    Returns: dict of POS features.
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    total = len(pos_tags) if pos_tags else 1
    pos_counts = {}
    for _, tag in pos_tags:
        pos_counts[tag] = pos_counts.get(tag, 0) + 1

    # Normalize counts by total token count
    pos_features = {f"POS_{tag}": count / total for tag, count in pos_counts.items()}
    return pos_features


def extract_hesitation_features(text):
    """
    Count occurrences of hesitation/filler words.
    """
    hesitations = ["uh", "um", "erm", "ah", "you know", "like", "mhm"]
    text_lower = text.lower()
    counts = {f"hes_{h}": len(re.findall(rf"\b{re.escape(h)}\b", text_lower))
              for h in hesitations}
    return counts


# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("preprocessed_data_cleaned.csv")  # Already cleaned_text + label + id
if "cleaned_text" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'cleaned_text' and 'label' columns.")

X_text = df["cleaned_text"].astype(str)
y_text = df["label"].astype(int)

# -----------------------
# POS & Hesitation Features
# -----------------------
print("🔍 Extracting POS & hesitation features...")
pos_feature_list = []
hes_feature_list = []

for txt in X_text:
    pos_feature_list.append(extract_pos_features(txt))
    hes_feature_list.append(extract_hesitation_features(txt))

df_pos = pd.DataFrame(pos_feature_list).fillna(0)
df_hes = pd.DataFrame(hes_feature_list).fillna(0)

# Keep same row order
extra_features = pd.concat([df_pos, df_hes], axis=1).to_numpy(dtype=np.float32)

# -----------------------
# TF-IDF Features
# -----------------------
print("🔍 Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(X_text)

# -----------------------
# Combine Features
# -----------------------
X_combined = hstack([
    csr_matrix(X_tfidf, dtype=np.float32),
    csr_matrix(extra_features, dtype=np.float32)
])

print(f"TF-IDF shape: {X_tfidf.shape}")
print(f"Extra features shape: {extra_features.shape}")
print(f"Combined shape: {X_combined.shape}")

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_text, test_size=0.2, stratify=y_text, random_state=42
)

# -----------------------
# SVM Training (best C=1)
# -----------------------
print("🚀 Training SVM with C=1...")
svm_model = LinearSVC(C=1, class_weight="balanced", max_iter=5000)
svm_model.fit(X_train, y_train)

# -----------------------
# Evaluation
# -----------------------
y_pred = svm_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nAccuracy: {acc:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")

# -----------------------
# Top Words
# -----------------------
feature_names = vectorizer.get_feature_names_out()
coefs = svm_model.coef_[0]

# Only show top words from TF-IDF part (exclude extra features)
top_class1_idx = np.argsort(coefs[:len(feature_names)])[-10:][::-1]
top_class0_idx = np.argsort(coefs[:len(feature_names)])[:10]

print("\nTop words for class 1 (Dementia):")
for idx in top_class1_idx:
    print(feature_names[idx])

print("\nTop words for class 0 (Control):")
for idx in top_class0_idx:
    print(feature_names[idx])
