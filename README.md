# 🧠 Early Dementia Detection using Machine Learning

An end-to-end machine learning project that predicts the likelihood of early dementia using cognitive, linguistic, and speech-derived features. The system leverages feature engineering, hyperparameter tuning, and ensemble learning techniques to improve prediction accuracy and robustness.

---

## 📌 Project Overview

Early diagnosis of dementia can significantly improve treatment planning and patient outcomes. This project applies machine learning algorithms to analyze linguistic and speech-based patterns that may indicate early cognitive decline.

The complete pipeline includes data preprocessing, feature engineering, model training, hyperparameter optimization, ensemble learning, and performance evaluation.

---

## 🚀 Features

- ✅ End-to-End Machine Learning Pipeline
- ✅ Advanced Linguistic & Speech Feature Engineering
- ✅ Data Preprocessing and Feature Selection
- ✅ Hyperparameter Optimization
- ✅ Multiple Machine Learning Models
- ✅ Ensemble Voting and Stacking Classifiers
- ✅ Modular and Scalable Codebase
- ✅ Real-world Inspired Predictive System

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning Techniques
- Feature Engineering
- Feature Selection
- Hyperparameter Tuning
- Ensemble Learning
- Model Stacking
- Voting Classifier

---

## 📂 Project Workflow

```
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Feature Engineering
   │
   ▼
Train-Test Split
   │
   ▼
Model Training
   │
   ├── Logistic Regression
   ├── Lasso Regression
   ├── Random Forest
   ├── Support Vector Machine
   └── XGBoost
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Ensemble Learning
   │
   ├── Voting Classifier
   └── Stacking Classifier
   │
   ▼
Performance Evaluation
```

---

## 🤖 Machine Learning Models

The project evaluates and compares the performance of several supervised learning algorithms:

- Logistic Regression (Hyperparameter Tuned)
- Lasso Regression
- Random Forest Classifier
- Support Vector Machine (RBF Kernel)
- XGBoost Classifier

---

## 🔥 Ensemble Learning

To improve prediction performance, two ensemble techniques were implemented:

### ✅ Voting Classifier

Combines predictions from:

- Lasso Regression
- Random Forest
- Support Vector Machine

### ✅ Stacking Classifier

Uses multiple base learners with a meta-model:

- XGBoost
- Random Forest
- Support Vector Machine

---

## 📊 Model Performance

The optimized ensemble voting classifier achieved approximately:

| Metric | Score |
|---------|------:|
| Accuracy | **~91%** |

> **Note:** Performance may vary depending on dataset splits and preprocessing techniques.

---

## 💡 Why Ensemble Learning?

Instead of relying on a single machine learning model, ensemble learning combines multiple strong learners to achieve better predictive performance.

### Benefits

- ✔ Reduces overfitting
- ✔ Improves prediction stability
- ✔ Better generalization on unseen data
- ✔ Higher overall accuracy
- ✔ More robust predictions

---


## 🎯 Future Improvements

- Deep Learning-based Models
- Transformer-based Language Models (BERT, RoBERTa)
- Speech Signal Processing Features
- Explainable AI (SHAP/LIME)
- Web-based Prediction Dashboard
- Real-time Clinical Decision Support

---

## 🤝 Contributing

Contributions are welcome!

If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Manvi Dhamija**

If you found this project useful, consider giving it a ⭐ on GitHub!
