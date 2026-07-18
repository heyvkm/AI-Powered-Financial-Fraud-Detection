# 🛡️ AI-Powered Financial Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.2-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered Financial Fraud Detection System that predicts whether a financial transaction is **Legitimate** or **Fraudulent** in real time using a **Balanced Random Forest** classifier trained on the **PaySim Mobile Money Transaction Dataset**.

The application includes automated feature engineering, pre-model input validation, fraud probability estimation, risk level assessment, plain-language prediction explanations, and a Streamlit dashboard with light/dark mode.

---

# 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Project Objectives](#-project-objectives)
- [Features](#-features)
- [Dataset](#-dataset)
- [Machine Learning Model](#-machine-learning-model)
- [Model Performance](#-model-performance)
- [Custom Feature Engineering](#-custom-feature-engineering)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Project Structure](#-project-structure)
- [Prediction Pipeline](#-prediction-pipeline)
- [Risk Levels](#-risk-levels)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Application Screenshots](#-application-screenshots)
- [Feature Importance](#-feature-importance)
- [Known Limitations](#-known-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

# 🎯 Project Overview

Financial fraud is one of the biggest challenges in digital banking and online payment systems. Detecting fraudulent transactions accurately while minimizing false positives is critical for financial institutions.

This project uses Machine Learning to identify fraudulent mobile money transactions based on transaction behavior rather than fixed business rules.

The application performs:

- Real-time fraud prediction
- Automatic feature engineering
- Fraud probability estimation
- Risk level classification
- Plain-language prediction explanation
- Prediction logging
- Interactive dashboard visualization

---

# 🎯 Project Objectives

- Detect fraudulent financial transactions in real time.
- Handle a highly imbalanced fraud dataset (fraud is 0.13% of all transactions).
- Reduce false positives via a separate, deterministic input-validation layer.
- Improve fraud detection using custom feature engineering.
- Build an interactive Streamlit web application.
- Demonstrate an end-to-end Machine Learning workflow.

---

# 🚀 Features

- 🤖 Real-time fraud prediction
- 📊 Fraud probability estimation
- 🚦 Risk Level (LOW / MEDIUM / HIGH / CRITICAL)
- 📝 Automatic feature engineering
- ✅ Pre-model input validation (blocks prediction on internally inconsistent input)
- 💡 Plain-language risk factor breakdown
- 📈 Global feature importance visualization
- 📜 Prediction history logging (`prediction_logs/predictions.csv`)
- 🌙 Dark / Light mode toggle
- ℹ️ About panel with author links

---

# 📊 Dataset

**Dataset:** PaySim Mobile Money Transaction Dataset


| Item | Value |
|------|-------|
| Total Transactions | 6,362,620 |
| Legitimate Transactions | 6,354,407 |
| Fraud Transactions | 8,213 |
| Fraud Rate | 0.13% |
| Target Variable | `isFraud` |
| Problem Type | Binary Classification |

You can download it from Kaggle:
🔗 https://www.kaggle.com/datasets/ealaxi/paysim1

---

# 🤖 Machine Learning Model

The training notebook ([`Financial_Fraud_Detection_System.ipynb`](Financial_Fraud_Detection_System.ipynb)) compares multiple algorithms, both as-is and with `class_weight="balanced"` to handle the extreme class imbalance.

### Models Evaluated

- Logistic Regression (baseline + balanced)
- Decision Tree (baseline + balanced)
- Random Forest (baseline + balanced)
- **Balanced Random Forest** ✅ (Final Model)

The final deployed model:

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

SMOTE and `RandomizedSearchCV` hyperparameter tuning were considered and deliberately not used — the balanced Random Forest already performed excellently, and the dataset's size (6.3M rows) made tuning impractically memory-heavy for the marginal gain expected.

---

# 🏆 Model Performance

Measured on a held-out 20% test split (stratified):

| Metric | Score |
|---------|-------|
| Algorithm | Balanced Random Forest |
| Features | 21 |
| Trees | 100 |
| Accuracy | **99.9997%** |
| Precision | **100.00%** |
| Recall | **99.76%** |
| F1 Score | **99.88%** |
| ROC-AUC Score | **99.88%** |

---

# 💡 Custom Feature Engineering

To improve fraud detection performance, 10 behavioral features are engineered from the 6 raw transaction fields before every prediction (see [`utils/feature_engineering.py`](utils/feature_engineering.py)):

- `hour`, `day` — derived from PaySim's `step` (hours since simulation start, 1–743)
- `sender_balance_change`, `receiver_balance_change`
- `amount_balance_ratio`
- `account_emptied` — sender's new balance hit exactly zero
- `large_transaction` — amount above the 95th percentile of the training data
- `sender_zero_balance`, `receiver_zero_balance`
- `dest_not_credited` — see below
- `isFlaggedFraud` — PaySim's own system flag, always `0` here (not derivable from a real-time user-submitted transaction)

### ⭐ `dest_not_credited` — the key custom feature

The original model missed a strong fraud signal: when a receiver's balance doesn't move at all despite a nonzero transaction amount, that pattern is fraud in **over 97%** of matching `TRANSFER` cases in the training data — a signal the raw balance columns alone didn't make explicit to the model.

```python
dest_not_credited = int(
    (newbalanceDest - oldbalanceDest) == 0
    and amount > 0
)
```

This check is **not** restricted to `TRANSFER` — it's computed for every transaction type, since the model itself learns from the full picture (type is captured separately via one-hot encoding). Restricting it upfront would throw away information the model can otherwise use.

The final model uses **21 input features**: 6 raw + 10 engineered (`isFlaggedFraud` counted here) + 4 one-hot transaction-type columns (`CASH_IN` is the dropped reference category).

---

# 🔄 Machine Learning Workflow

```text
Problem Understanding
        ↓
Data Collection & Understanding
        ↓
Data Quality Assessment
        ↓
Exploratory Data Analysis (EDA)
        ↓
Feature Engineering
        ↓
Feature Selection
        ↓
Data Preparation (split, encode)
        ↓
Baseline Model Development
        ↓
Model Evaluation
        ↓
Handling Class Imbalance
        ↓
Final Model Selection
        ↓
Model Saving
        ↓
Streamlit Web Application (this repo)
```

---

# ⚙️ Prediction Pipeline

Every prediction follows the exact preprocessing steps used during training.

### 1️⃣ Input Validation — [`utils/helpers.py`](utils/helpers.py)

`validate_transaction()` runs **before** the model ever sees the input:

- All fields must be non-negative
- Sender's new balance can't exceed the old balance (except `CASH_IN`, where the sender is receiving money)
- The sender's actual balance change must match the stated amount (₹1 tolerance)

If validation fails, the app shows the specific mismatch and **stops** — no verdict, probability, or risk breakdown is shown for internally inconsistent data.

Deliberately **not** checked: receiver-side consistency. PaySim has many legitimate transactions (e.g. `PAYMENT` to merchants) where the receiver's balance doesn't move by the full amount, and the model relies on that exact pattern (`dest_not_credited`) as a real fraud signal — enforcing it here would reject valid inputs the model was trained to handle.

### 2️⃣ Feature Engineering — [`utils/feature_engineering.py`](utils/feature_engineering.py)

The 10 engineered features described above are computed from the 6 raw inputs.

### 3️⃣ Encoding

Transaction type is one-hot encoded using the saved `OneHotEncoder` (`models/onehot_encoder.pkl`).

### 4️⃣ Prediction — [`utils/predictor.py`](utils/predictor.py)

All features are reordered to match `model.feature_names_in_` (pulled live from the model object, never hardcoded) and passed to the Balanced Random Forest.

Output:

- Fraud prediction (0/1)
- Fraud probability
- Risk level

### 5️⃣ Risk Factor Breakdown — [`utils/helpers.py`](utils/helpers.py)

`get_risk_factors()` turns the engineered features into plain-language bullets (e.g. *"Receiver balance did not change despite a ₹50,000 transfer..."*). This is rule-based logic layered on top of the real prediction — not a second model.

### 6️⃣ Prediction Logging — [`utils/logger.py`](utils/logger.py)

Every successful prediction (i.e. one that passed validation) is appended to:

```
prediction_logs/predictions.csv
```

---

# 🚦 Risk Levels

| Fraud Probability | Risk Level |
|------------------|------------|
| < 20% | 🟢 LOW |
| 20% – < 50% | 🟡 MEDIUM |
| 50% – < 80% | 🟠 HIGH |
| ≥ 80% | 🔴 CRITICAL |

---

# 🛠️ Technologies Used

## Programming Language

- Python 3.13

## Machine Learning

- scikit-learn (`RandomForestClassifier`, `OneHotEncoder`)
- pandas, NumPy
- joblib (model serialization)

## Visualization (training notebook only — not used in the deployed app)

- matplotlib, seaborn

## Deployment

- Streamlit

---

# 📂 Project Structure

```text
.
├── app.py                       # Streamlit UI — all business logic lives in utils/
├── config.py                    # Model/encoder paths, TRAINING_THRESHOLD constant
├── requirements.txt
├── README.md
├── LICENSE
│
├── models/
│   ├── fraud_detection_model.pkl   # Balanced Random Forest (21 features)
│   └── onehot_encoder.pkl          # OneHotEncoder fit on the 'type' column
│
├── utils/
│   ├── loader.py                # Cached model/encoder loading
│   ├── feature_engineering.py   # engineer_features() — the 10 derived features
│   ├── predictor.py             # predict() — full inference pipeline
│   ├── helpers.py               # validate_transaction(), get_risk_factors()
│   └── logger.py                # log_prediction() — writes prediction_logs/predictions.csv
│
├── assets/
│   └── style.css                # Static CSS (all colors via CSS variables for dark/light mode)
│
├── prediction_logs/
│   └── predictions.csv          # Append-only log of completed predictions
│
├── frauddetection.ipynb         # Training notebook (EDA → feature engineering → model selection)
└── Financial_datasets_log.csv   # PaySim training data (gitignored — not committed, ~490MB)
```

---

# ▶️ Installation

## Get the project

Copy or clone this folder to your machine, then from inside it:

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

The app resolves `models/fraud_detection_model.pkl` and `models/onehot_encoder.pkl` relative to its own file location (`config.py` uses `Path(__file__)`), so it works regardless of the directory you launch it from.

> **scikit-learn version:** the model was trained on `scikit-learn==1.7.2`, pinned in `requirements.txt`. Loading it with a different version still works but throws an `InconsistentVersionWarning`.

---

# 📸 Application Screenshots

_Not yet added — drop images into `assets/screenshots/` and reference them here, e.g.:_

```
assets/screenshots/dashboard.png
assets/screenshots/fraud_result.png
assets/screenshots/legitimate_result.png
```

---

# 📈 Feature Importance

The most influential features learned by the model, in order:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `amount_balance_ratio` | 0.250 |
| 2 | `sender_balance_change` | 0.167 |
| 3 | `oldbalanceOrg` | 0.129 |
| 4 | `newbalanceOrig` | 0.102 |
| 5 | `account_emptied` | 0.086 |
| 6 | `amount` | 0.060 |
| 7 | `type_TRANSFER` | 0.028 |
| 8 | `large_transaction` | 0.024 |
| 9 | `receiver_balance_change` | 0.024 |
| 10 | `dest_not_credited` | 0.006 |

`dest_not_credited` ranks low in *overall* importance because the pattern it captures is rare (~0.8% of transfers) — but within that narrow slice, it's decisive: recall on transactions matching this pattern is 100%, up from the model missing it entirely before the feature was added.

---

# ⚠️ Known Limitations

- **Not an arithmetic checker on its own.** The model learns statistical patterns; it doesn't verify balances add up. That's what the separate validation layer (Pipeline step 1) is for.
- **Receiver-side blind spot is intentional**, not a bug — see the `dest_not_credited` section above.
- **`isFlaggedFraud` is always `0`** in this app — it was PaySim's own internal simulation flag, not derivable from a real-time transaction.
- **Amount alone is a weak signal.** The model relies far more on *behavioral* patterns (account fully drained, receiver not credited) than on transaction size — a ₹500 transaction can be flagged Fraud and a ₹2 crore transaction can be Legitimate, depending on those patterns.
- **Trained on synthetic data.** PaySim is a simulator; real-world banking data would likely require retraining and re-validation before production use.

---

# 🚀 Future Improvements

- Explainable AI using SHAP for true per-prediction feature attribution
- REST API (FastAPI) for programmatic access
- Database-backed prediction history instead of a flat CSV
- Batch/CSV upload for checking multiple transactions at once
- Dockerized deployment
- Cloud deployment (AWS / Azure / GCP)

---

# 👨‍💻 Author

**Vishal Kumar Maurya**

🎓 B.Tech Information Technology
🏫 Dr. Ambedkar Institute of Technology for Handicapped (AITH), Kanpur
💼 IBM PBEL Internship Project

### Connect with Me

- 🔗 GitHub: [github.com/heyvkm](https://github.com/heyvkm)
- 💼 LinkedIn: [linkedin.com/in/vishalkmaury](https://www.linkedin.com/in/vishalkmaury/)

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐.

---

## 📄 License

Released under the [MIT License](LICENSE) — free to fork, modify, and use for learning.
