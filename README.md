
# 🕵️‍♂️ E-Commerce Fraud Detection Project

This project builds machine learning models to detect fraudulent transactions using multiple datasets that capture user behaviors, transaction metadata, and geolocation. The workflow covers data preprocessing, feature engineering, model training, evaluation, and model explainability using SHAP.

---

## 📦 Datasets Used

1. **Fraud_Data.csv**  
   Transaction-level records including:
   - `signup_time`, `purchase_time`
   - User & transaction details (`device_id`, `source`, `browser`, etc.)
   - `fraud` (target label: 1 for fraud, 0 for legit)

2. **IpAddress_to_Country.csv**  
   IP-to-country mapping based on user session IPs.

3. **creditcard.csv**  
   Kaggle dataset for credit card fraud detection (anonymized numerical features and `Class` target).

---

## 🛠️ Key Features & Engineering

- **Time-Based Features:**
  - `hour_of_day` of purchase
  - `day_of_week` of purchase
  - `time_since_signup` (in hours)

- **Transaction Frequency/Velocity:**
  - Number of transactions per user in a rolling window
  - Time between transactions

- **Categorical Encoding:**
  - Label encoding: `source`, `browser`, `sex`
  - Dropped high-cardinality column `device_id` (>130K unique values)

- **Datetime Handling:**
  - Extracted features from `signup_time` and `purchase_time`
  - Dropped raw datetime columns after transformation

---

## 🧠 Machine Learning Models

### ✅ Task 2: Model Training & Evaluation

Models were trained on both **e-commerce** and **credit card** datasets using:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**

### 🧪 Evaluation Metrics
Each model was evaluated using:
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- Cross-Validation (Stratified K-Fold)

Hyperparameter tuning was applied using `GridSearchCV`.

📁 Model artifacts are saved in `models/` as `.pkl` files.

---

## 📊 Task 3: SHAP Explainability

To understand model predictions, SHAP (SHapley Additive exPlanations) was used for:

- **XGBoost**: Tree-based SHAP explainers
- **Logistic Regression**: Linear SHAP explainers

### 🔍 Visuals Generated:
- SHAP **Summary Plot**: Feature importance
- SHAP **Force Plot**: Local explanation of a specific prediction

📁 Saved under `reports/figures/`:
```
shap_summary_xgb_ecommerce.png  
shap_force_xgb_ecommerce.png  
shap_summary_logreg_creditcard.png  
shap_force_logreg_creditcard.png  
...
```

---

## 📁 Project Structure

| File / Folder | Description |
|---------------|-------------|
| `notebooks/EDA_and_Preprocessing.ipynb` | Initial EDA and feature generation |
| `notebooks/Model_Training_and_Evaluation.ipynb` | Model training, evaluation & tuning |
| `notebooks/SHAP_Interpretation.ipynb` | SHAP model interpretation |
| `src/transform_preprocessing.py` | Custom preprocessing utilities |
| `src/model_training.py` | ML training pipeline |
| `src/shap_interpretation.py` | SHAP-based interpretability scripts |
| `models/` | Trained model artifacts |
| `data/processed/` | Transformed training datasets |
| `reports/figures/` | SHAP plots for model interpretability |
| `.venv/` | Python virtual environment (excluded from Git) |

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/tsegabogale24/-detection-of-fraud-cases-for-e-commerce
cd -detection-of-fraud-cases-for-e-commerce
```

### 2. Set up the environment
```bash
python -m venv .venv
source .venv/Scripts/activate  # or .venv/bin/activate on Unix
pip install -r requirements.txt
```

### 3. Run notebooks
Open Jupyter and run:
- `01_data_preprocessig.ipynb`
- `task2_model_training.ipynb`
- `03_model_explainability.ipynb`

---

## 🚀 Next Steps (Coming Soon)

- Deploy model using FastAPI
- Streamlit dashboard for fraud alert visualization
- AutoML comparison using PyCaret
