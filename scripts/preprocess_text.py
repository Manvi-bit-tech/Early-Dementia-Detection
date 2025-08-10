# scripts/preprocess_text.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_text(text):
    """Lowercase, remove punctuation, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    return " ".join(tokens)

def load_text_data(path='preprocessed_data.csv'):
    """Load and clean text dataset, or return if already cleaned."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # normalize names

    # If already cleaned
    if {"id", "cleaned_text", "label"}.issubset(df.columns):
        return df[["id", "cleaned_text", "label"]]

    # Otherwise, look for raw text column
    possible_text_cols = ["Processed_Text", "processed_text", "Text", "text"]
    text_col = next((col for col in possible_text_cols if col in df.columns), None)
    if not text_col:
        raise KeyError(f"❌ No text column found. Columns: {list(df.columns)}")

    possible_label_cols = ["Label", "label"]
    label_col = next((col for col in possible_label_cols if col in df.columns), None)
    if not label_col:
        raise KeyError(f"❌ No label column found. Columns: {list(df.columns)}")

    df["cleaned_text"] = df[text_col].apply(clean_text)
    df = df.rename(columns={label_col: "label"})
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)

    return df[["id", "cleaned_text", "label"]]

if __name__ == "__main__":
    input_path = "preprocessed_data.csv"
    output_path = "preprocessed_data_cleaned.csv"
    df_clean = load_text_data(input_path)
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Saved cleaned dataset to {output_path} with {df_clean.shape[0]} samples")
