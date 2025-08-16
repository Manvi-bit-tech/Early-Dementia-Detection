# app.py
import streamlit as st
import joblib
import librosa
import numpy as np
import re
import nltk
from scipy.sparse import hstack, csr_matrix

# --------------------------
# Load trained models
# --------------------------
text_bundle = joblib.load("text_model.pkl")
audio_bundle = joblib.load("audio_model.pkl")

text_model = text_bundle["model"]
text_vectorizer = text_bundle["vectorizer"]
extra_columns = text_bundle.get("extra_columns", [])

audio_model = audio_bundle["model"]
audio_scaler = audio_bundle["scaler"]
audio_pca = audio_bundle.get("pca", None)

# --------------------------
# UI Setup
# --------------------------
st.set_page_config(page_title="🧠 Dementia Detection", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
    .stTabs [role="tablist"] {
        justify-content: center;
    }
    .stTabs [role="tab"] {
        font-size: 18px;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 10px;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .success {
        background-color: #e0f7e9;
        border-left: 5px solid #28a745;
    }
    .danger {
        background-color: #fdecea;
        border-left: 5px solid #dc3545;
    }
    .neutral {
        background-color: #f1f3f5;
        border-left: 5px solid #6c757d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Early Dementia Detection")
st.caption("AI-powered tool to analyze **text** or **audio** for early dementia signs.")

# --------------------------
# Helpers
# --------------------------
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

def sigmoid(x): return 1 / (1 + np.exp(-x))

def extract_pos_features(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    total = len(pos_tags) if pos_tags else 1
    counts = {}
    for _, tag in pos_tags:
        counts[tag] = counts.get(tag, 0) + 1
    return {f"POS_{t}": c/total for t, c in counts.items()}

def extract_hesitation_features(text):
    hes = ["uh","um","erm","ah","you know","like","mhm"]
    txt = text.lower()
    return {f"hes_{h}": len(re.findall(rf"\\b{re.escape(h)}\\b", txt)) for h in hes}

def prepare_text_features(text):
    X_tfidf = text_vectorizer.transform([text])
    pos_feats = extract_pos_features(text)
    hes_feats = extract_hesitation_features(text)
    feats = {**pos_feats, **hes_feats}
    for col in extra_columns:
        if col not in feats:
            feats[col] = 0
    X_extra = np.array([feats[col] for col in extra_columns]).reshape(1, -1)
    return hstack([X_tfidf, csr_matrix(X_extra)])

def extract_audio_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1),
        np.mean(mel_db, axis=1), np.std(mel_db, axis=1)
    ])

def prepare_audio_features(file):
    y, sr = librosa.load(file, sr=None)
    feats = extract_audio_features(y, sr)
    X_scaled = audio_scaler.transform([feats])
    if audio_pca:
        X_scaled = audio_pca.transform(X_scaled)
    return X_scaled

# --------------------------
# Tabs for Modes
# --------------------------
tab1, tab2= st.tabs(["✍️ Text", "🎵 Audio"])

# ---- Text Tab ----
with tab1:
    st.subheader("✍️ Text-based Prediction")
    user_input = st.text_area("Paste patient transcript or text below:")
    if st.button("Analyze Text", key="text_btn"):
        if user_input.strip():
            X_text = prepare_text_features(user_input)
            pred = text_model.predict(X_text)[0]
            raw_score = text_model.decision_function(X_text)[0]
            prob = sigmoid(raw_score)

            with st.container():
                if pred == 1:
                    st.markdown(
                        f"<div class='result-card danger'><h3 style=\"color: blue;\">🧠 Dementia Detected</h3><p style=\"color: blue;\">Confidence: {prob:.2f}</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='result-card success'><h3 style=\"color: blue;\">✅ Control / No Dementia</h3><p style=\"color: blue;\">Confidence: {1-prob:.2f}</p></div>",
                        unsafe_allow_html=True,
                    )
                st.progress(float(prob if pred == 1 else 1-prob))
        else:
            st.warning("⚠️ Please enter some text.")

# ---- Audio Tab ----
with tab2:
    st.subheader("🎵 Audio-based Prediction")
    uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav","mp3"], key="audio_uploader")
    if uploaded_file and st.button("Analyze Audio", key="audio_btn"):
        try:
            X_audio = prepare_audio_features(uploaded_file)
            pred = audio_model.predict(X_audio)[0]
            prob = audio_model.predict_proba(X_audio)[0][1]

            with st.container():
                if pred == 1:
                    st.markdown(
                        f"<div class='result-card danger'><h3 style=\"color: blue;\">🧠 Dementia Detected</h3><p style=\"color: blue;\">Confidence: {prob:.2f}</p></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div class='result-card success'><h3 style=\"color: blue;\">✅ Control / No Dementia </h3><p style=\"color: blue;\">Confidence: {1-prob:.2f}</p></div>",
                        unsafe_allow_html=True,
                    )
                st.progress(float(prob if pred == 1 else 1-prob))
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---- Fusion Tab ----
# with tab3:
 #   st.subheader("🔗 Fusion Prediction (Coming Soon)")
  #  st.info("This section will combine text + audio models for a stronger decision. 🚀")
