from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
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


def get_preprocessor(numeric_cols, categorical_cols, scaling='standard'):
    """Returns a preprocessor that scales numeric and encodes categorical features."""
    scaler = StandardScaler() if scaling == 'standard' else MinMaxScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    return preprocessor


def transform_data(df, target_col, numeric_cols, categorical_cols,
                   imbalance_strategy='smote', scaling='standard', test_size=0.2, random_state=42):
    """Prepares data: splits, encodes, balances, and returns transformed arrays and preprocessor."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Explicitly drop unwanted columns
    to_drop = []
    if 'signup_time' in X.columns:
        to_drop.append('signup_time')
    if 'purchase_time' in X.columns:
        to_drop.append('purchase_time')
    if 'device_id' in X.columns:
        to_drop.append('device_id')

    if to_drop:
        print(f"Dropping columns: {to_drop}")
        X = X.drop(columns=to_drop)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # Create preprocessor
    preprocessor = get_preprocessor(numeric_cols, categorical_cols, scaling=scaling)

    # Transform features
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Balance the classes
    X_train_balanced, y_train_balanced = handle_class_imbalance(
        X_train_encoded, y_train, strategy=imbalance_strategy, random_state=random_state
    )

    return X_train_balanced, X_test_encoded, y_train_balanced, y_test, preprocessor
