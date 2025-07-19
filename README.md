# 🕵️‍♂️ E-Commerce Fraud Detection Project

This project aims to build machine learning models to detect fraudulent transactions using multiple datasets that capture user behaviors, transaction metadata, and geolocation. The workflow includes data preprocessing, feature engineering, model training, and evaluation.

---

## 📦 Datasets Used

1. **Fraud_Data.csv**  
   Contains transaction-level records including:
   - `signup_time`, `purchase_time`
   - User & transaction details (`device_id`, `source`, `browser`, etc.)
   - `fraud` (target label: 1 for fraud, 0 for legit)

2. **IpAddress_to_Country.csv**  
   IP-to-country mapping based on user session IPs.

3. **creditcard.csv**  
   Standard credit card fraud detection dataset (from Kaggle).

---

## 🛠️ Key Features & Engineering

- **Time-Based Features:**
  - `hour_of_day` of purchase
  - `day_of_week` of purchase
  - `time_since_signup` (in hours)
  
- **Transaction Frequency/Velocity:**
  - Number of transactions per user in the last X hours
  - Average time between purchases

- **Categorical Encoding:**
  - Label encoding for low-cardinality columns (`source`, `browser`, `sex`)
  - Dropped high-cardinality column `device_id` (137k+ unique values)

- **Datetime Handling:**
  - Extracted meaningful numeric features from datetime columns
  - Dropped raw datetime fields before scaling

---

## 🧠 Models (Coming Soon)

Models will include:
- Logistic Regression
- Random Forest
- XGBoost
- Resampling Techniques (SMOTE, ADASYN)

---

## 🧪 Notebooks & Scripts

| File | Description |
|------|-------------|
| `notebooks/EDA_and_Preprocessing.ipynb` | Exploratory Data Analysis and Feature Engineering |
| `src/transform_preprocessing.py` | Utility functions for encoding, scaling, and transforming data |
| `data/` | Folder for input datasets |
| `.venv/` | Python virtual environment (excluded in `.gitignore`) |

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/tsegabogale24/-detection-of-fraud-cases-for-e-commerce
cd -detection-of-fraud-cases-for-e-commerce
