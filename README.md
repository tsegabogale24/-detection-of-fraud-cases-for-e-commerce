# 🕵️‍♂️ E-Commerce & Credit Card Fraud Detection

This project is an end-to-end machine learning pipeline designed to detect fraudulent transactions using two datasets: a synthetic **e-commerce fraud dataset** and a real-world **credit card fraud dataset** from Kaggle. We built, evaluated, and interpreted models with modern explainability tools like SHAP to uncover patterns in fraud behavior.

---

## 📊 Datasets Used

### 1. `Fraud_Data.csv` (E-Commerce Fraud Dataset)
- Includes user-level transaction metadata like:
  - `purchase_time`, `signup_time`
  - User attributes: `browser`, `sex`, `source`, `age`, `device_id`
  - IP-related features
  - Target: `fraud` (1 = fraud, 0 = legit)

### 2. `IpAddress_to_Country.csv`
- Maps user IPs to countries for geolocation-based features.

### 3. `creditcard.csv`
- Standard dataset for credit card fraud detection from Kaggle.
- 284,807 transactions with 492 fraud cases (high class imbalance).

---

## 🛠️ Feature Engineering

### ✅ For Both Datasets:
- **Time-based features**:
  - `hour_of_day`, `day_of_week`
  - `time_since_signup` (e-commerce)
- **Categorical encoding**:
  - Label encoding: `source`, `browser`, `sex`
- **Dropped high-cardinality ID columns**:
  - `device_id`, `user_id`
- **Geolocation (E-Commerce only)**:
  - Country derived from IP-to-country mapping.

---

## 🤖 Models Trained

We applied both simple and advanced models with proper handling of class imbalance.

| Dataset | Models | Resampling | Evaluation |
|--------|--------|------------|------------|
| **E-Commerce** | Logistic Regression, Random Forest, XGBoost | SMOTE | ROC-AUC, Confusion Matrix |
| **Credit Card** | Logistic Regression, Random Forest, XGBoost | None (already imbalanced) | ROC-AUC, F1, Precision, Recall |

---

## 🔍 Model Interpretability with SHAP

### ✅ SHAP Visualizations (Global + Local):
We used **SHAP** (SHapley Additive exPlanations) to interpret both XGBoost and Logistic Regression models.

#### 📈 SHAP Summary Plots
- Highlight top features driving fraud predictions.
- In e-commerce data, `time_since_signup`, `hour_of_day`, and `country` were strong indicators.
- In credit card data, anonymized features like `V14`, `V10`, and `V17` were most important.

#### 🎯 SHAP Force Plots
- Explained **individual predictions** for specific transactions.
- Great for model transparency and trust.

All SHAP plots are saved in `reports/figures`.

---

## 📁 Project Structure

├── data/
│ ├── raw/
│ ├── processed/
│ └── external/
├── notebooks/
│ ├── EDA_and_Preprocessing.ipynb
│ ├── Model_Training.ipynb
│ └── SHAP_Interpretation.ipynb
├── src/
│ ├── transform_preprocessing.py
│ ├── train_models.py
│ ├── shap_interpretation.py
├── reports/
│ └── figures/ ← SHAP plots, confusion matrices, etc.
├── models/
│ └── *.pkl (saved models)
├── README.md
└── requirements.txt



---

## 🧪 How to Run

### 1. Clone the repo
```
git clone https://github.com/your-username/detection-of-fraud-cases-for-e-commerce.git
cd detection-of-fraud-cases-for-e-commerce
2. Create and activate virtual environment

python -m venv .venv
source .venv/Scripts/activate  # On Linux: source .venv/bin/activate
3. Install dependencies

pip install -r requirements.txt
4. Run Notebooks or Scripts
All code is modularized and runnable from notebooks/ or by importing from src/.

📊 Evaluation Metrics
Metric	Description
ROC-AUC	Measures discrimination power between classes
F1-Score	Balance between precision and recall
Precision	True Positives / (True Positives + False Positives)
Recall	True Positives / (True Positives + False Negatives)

📝 Final Report
A detailed Medium-style report is available in the reports/ folder or hosted online.

It covers:

Problem background

Dataset description

Feature engineering

Model comparisons

SHAP explainability

Recommendations

💡 Key Takeaways
XGBoost consistently outperformed Logistic Regression and Random Forest.

SHAP was critical for uncovering the most important fraud indicators.

Class imbalance was successfully handled using SMOTE and evaluation metrics beyond accuracy.

The pipeline is modular and can be extended to real-world production settings.

🙌 Acknowledgements
Kaggle for the credit card fraud dataset.

The creators of the synthetic e-commerce fraud dataset.

SHAP by Scott Lundberg.

📧 Contact
Tsega Bogale
GitHub: @tsegabogale24
Email: tsegabogale24@gmail.com
