# Early-Dementia-Detection
# 🧠 Early Dementia Detection (Text + Audio ML Models)

This project is an **Early Dementia Detection system** built with **Machine Learning** and deployed via **Streamlit**.  
It supports text-based and audio-based predictions, with a future plan to add **fusion inference**.

---

## 🚀 Features
- **Text Prediction**  
  Paste cleaned transcripts → model predicts dementia vs control.
- **Audio Prediction**  
  Upload `.wav` or `.mp3` file → model extracts features and predicts.
- **Fusion (Coming Soon)**  
  Future-proof design for combining predictions from text + audio.

---

## 📂 Project Structure
# 🧠 Early Dementia Detection (Text + Audio ML Models)

This project is an **Early Dementia Detection system** built with **Machine Learning** and deployed via **Streamlit**.  
It supports text-based and audio-based predictions, with a future plan to add **fusion inference**.

---

## 🚀 Features
- **Text Prediction**  
  Paste cleaned transcripts → model predicts dementia vs control.
- **Audio Prediction**  
  Upload `.wav` or `.mp3` file → model extracts features and predicts.
- **Fusion (Coming Soon)**  
  Future-proof design for combining predictions from text + audio.

---

## 📂 Project Structure
Early-Dementia-Detection/
│── app.py # Streamlit app
│── requirements.txt # Dependencies
│── README.md # Project documentation
│── text_model.pkl # Trained text model
│── audio_model.pkl # Trained audio model
│── scripts/ # Training & utility scripts
│── data/ # (Optional) sample dataset


---

## ⚙️ Setup & Run Locally

### 1️⃣ Clone Repo
```bash
git clone https://github.com/Manvi-bit-tech/Early-Dementia-Detection.git
cd Early-Dementia-Detection

python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

pip install -r requirements.txt

streamlit run app.py

📊 Models

Text Model: Trained using TF-IDF + POS + Hesitation features with Linear SVM.

Audio Model: Trained using MFCC, deltas, spectral features with XGBoost.

Both models are saved as .pkl and loaded directly in the app.

👩‍💻 Author

Manvi Dhamija
B.Tech ECE | AI/ML Enthusiast | Data Science Learner