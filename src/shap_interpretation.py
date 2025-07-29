import shap
import pandas as pd
import matplotlib.pyplot as plt
import os

def explain_xgb_with_shap(model, X_train, feature_names, dataset_name="dataset", sample_index=0):
    """
    Generate SHAP summary and force plots for XGBoost models using the newer SHAP API.
    """
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    os.makedirs("../reports/figures", exist_ok=True)

    # Summary plot
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f"SHAP Summary Plot - XGBoost ({dataset_name})")
    plt.savefig(f"../reports/figures/shap_summary_xgb_{dataset_name}.png", bbox_inches="tight")
    plt.clf()

    # Force plot
    shap.plots.force(shap_values[sample_index], matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot - XGBoost ({dataset_name}) Sample {sample_index}")
    plt.savefig(f"../reports/figures/shap_force_xgb_{dataset_name}.png", bbox_inches="tight")
    plt.clf()

    return shap_values

def explain_logreg_with_shap(model, X_train, feature_names, dataset_name="dataset", sample_index=0):
    """
    Generate SHAP plots for Logistic Regression using the updated SHAP API.
    """
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    os.makedirs("../reports/figures", exist_ok=True)

    # Summary plot
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f"SHAP Summary Plot - Logistic Regression ({dataset_name})")
    plt.savefig(f"../reports/figures/shap_summary_logreg_{dataset_name}.png", bbox_inches="tight")
    plt.clf()

    # Force plot
    shap.plots.force(shap_values[sample_index], matplotlib=True, show=False)
    plt.title(f"SHAP Force Plot - Logistic Regression ({dataset_name}) Sample {sample_index}")
    plt.savefig(f"../reports/figures/shap_force_logreg_{dataset_name}.png", bbox_inches="tight")
    plt.clf()

    return shap_values
