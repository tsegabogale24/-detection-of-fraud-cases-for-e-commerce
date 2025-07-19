import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_class_distribution(df: pd.DataFrame, class_col: str, title: str = "Class Distribution"):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=class_col)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

def plot_numeric_distribution(df: pd.DataFrame, column: str, hue: str = None, bins: int = 30, title: str = None):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, hue=hue, bins=bins, kde=True)
    plt.title(title or f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_boxplot_by_class(df: pd.DataFrame, column: str, class_col: str):
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=class_col, y=column)
    plt.title(f"{column} by Class")
    plt.xlabel("Class")
    plt.ylabel(column)
    plt.show()

def plot_categorical_by_class(df: pd.DataFrame, column: str, class_col: str):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column, hue=class_col)
    plt.title(f"{column} Distribution by Class")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap"):
    plt.figure(figsize=(14, 10))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f")
    plt.title(title)
    plt.show()

def plot_top_n_countries(ip_df: pd.DataFrame, n: int = 10):
    plt.figure(figsize=(8, 4))
    country_counts = ip_df["country"].value_counts().nlargest(n)
    sns.barplot(x=country_counts.index, y=country_counts.values)
    plt.title(f"Top {n} Countries by IP Block Count")
    plt.xlabel("Country")
    plt.ylabel("IP Block Count")
    plt.xticks(rotation=45)
    plt.show()
