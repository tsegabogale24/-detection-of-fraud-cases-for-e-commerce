# src/features/engineering.py

import pandas as pd
import numpy as np
import ipaddress
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Univariate Plot
# ----------------------
def plot_univariate_distribution(df, column, title="", bins=30):
    plt.figure(figsize=(8, 4))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(title or f"Distribution of {column}")
    plt.show()

# ----------------------
# Bivariate Plot
# ----------------------
def plot_bivariate_boxplot(df, x, y, title=""):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[x], y=df[y])
    plt.title(title or f"{y} by {x}")
    plt.show()

# ----------------------
# IP Address Conversion
# ----------------------
def ip_to_int(ip_str):
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except:
        return np.nan

# ----------------------
# Merge IP to Country
# ----------------------
def merge_ip_country(fraud_df, ip_df):
    fraud_df["ip_address_int"] = fraud_df["ip_address"].apply(ip_to_int)

    def lookup_country(ip):
        match = ip_df[(ip_df["lower_bound_ip_address"] <= ip) & (ip_df["upper_bound_ip_address"] >= ip)]
        return match["country"].values[0] if not match.empty else "Unknown"

    fraud_df["country"] = fraud_df["ip_address_int"].apply(lookup_country)
    return fraud_df

