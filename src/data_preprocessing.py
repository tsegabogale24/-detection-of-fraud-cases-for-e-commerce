# task1_data_preprocessing.py (Modularized)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load Datasets
# -------------------------------
def load_datasets():
    fraud_df = pd.read_csv("../data/raw/Fraud_Data.csv")
    ip_df = pd.read_csv("../data/external/IpAddress_to_Country.csv")
    credit_df = pd.read_csv("../data/raw/creditcard.csv")
    return fraud_df, ip_df, credit_df

# -------------------------------
# 2. Initial Exploration
# -------------------------------
def explore_dataset(df, name="Dataset"):
    print(f"===== {name} Overview =====")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())
    print("\nClass Distribution:")
    if 'class' in df.columns:
        print(df['class'].value_counts(normalize=True))
    print("\n=============================\n")

# -------------------------------
# 3. Data Cleaning
# -------------------------------
def clean_data(df):
    df = df.drop_duplicates()
    df = df.fillna(method='ffill')
    return df

# -------------------------------
# 4. Convert Data Types
# -------------------------------
def convert_datatypes(fraud_df):
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
    return fraud_df

# -------------------------------
# 5. Plot Class Distribution
# -------------------------------
def plot_class_distribution(df, name):
    if 'class' in df.columns:
        sns.countplot(x='class', data=df)
        plt.title(f"Fraud Class Distribution in {name}")
        plt.show()
