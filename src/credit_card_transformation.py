from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd


def handle_class_imbalance(X, y, strategy='smote', random_state=42):
    """Handles class imbalance using SMOTE or RandomUnderSampler."""
    if strategy == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        raise ValueError("imbalance_strategy must be 'smote' or 'undersample'")

    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled


def get_numeric_preprocessor(numeric_cols, scaling='standard'):
    """Returns a preprocessor that scales numeric features only."""
    scaler = StandardScaler() if scaling == 'standard' else MinMaxScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_cols)
        ]
    )
    return preprocessor


def transform_creditcard_data(df, target_col,
                              numeric_cols, scaling='standard',
                              imbalance_strategy='smote', test_size=0.2, random_state=42):
    """Prepares credit card fraud data: splits, scales, balances, and returns transformed arrays and preprocessor."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # Create and apply scaler
    preprocessor = get_numeric_preprocessor(numeric_cols, scaling=scaling)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # Handle imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(
        X_train_scaled, y_train, strategy=imbalance_strategy, random_state=random_state
    )

    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, preprocessor
