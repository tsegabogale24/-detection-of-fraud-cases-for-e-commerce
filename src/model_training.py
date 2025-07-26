# train_fraud_and_credit.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def load_credit_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['Class'])
    y = df['Class']
    return X, y

def load_fraud_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.drop(columns=['class'])
    y_train = train_df['class']
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def balance_data(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

def train_models(X_train, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_model(model, X_test, y_test, dataset_name):
    y_pred = model.predict(X_test)
    print(f"\n--- {dataset_name} | {model.__class__.__name__} ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    print("PR AUC :", average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))

def run_credit_pipeline():
    X, y = load_credit_data("../data/raw/creditcard.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train, X_test = scale_features(X_train, X_test)
    X_train, y_train = balance_data(X_train, y_train)
    models = train_models(X_train, y_train)
    for model in models.values():
        evaluate_model(model, X_test, y_test, "CreditCard")

def run_fraud_pipeline():
    X_train, X_test, y_train, y_test = load_fraud_data(
        "../data/processed/transformed_train.csv",
        "../data/processed/transformed_test.csv"
    )
    X_train, X_test = scale_features(X_train, X_test)
    X_train, y_train = balance_data(X_train, y_train)
    models = train_models(X_train, y_train)
    for model in models.values():
        evaluate_model(model, X_test, y_test, "E-Commerce Fraud")

if __name__ == "__main__":
    run_credit_pipeline()
    run_fraud_pipeline()
