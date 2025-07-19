import pandas as pd

def preprocess_datetime_columns(df):
    """Ensure datetime columns are parsed correctly."""
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

def add_transaction_frequency(df):
    """Add transaction frequency per user."""
    freq_df = df.groupby('user_id').size().reset_index(name='transaction_count')
    df = df.merge(freq_df, on='user_id', how='left')
    return df

def add_transaction_velocity(df):
    """Compute user-level velocity = count / time span in hours, handle zero duration."""
    user_timespan = df.groupby('user_id')['purchase_time'].agg(['min', 'max'])
    user_timespan['duration_hours'] = (user_timespan['max'] - user_timespan['min']).dt.total_seconds() / 3600

    # Avoid division by zero
    user_timespan['duration_hours'] = user_timespan['duration_hours'].replace(0, 1e-6)

    user_transaction_count = df.groupby('user_id').size()
    user_timespan['velocity'] = user_transaction_count / user_timespan['duration_hours']
    velocity_df = user_timespan[['velocity']].reset_index()

    df = df.merge(velocity_df, on='user_id', how='left')
    return df


def add_time_features(df):
    """Extract hour_of_day, day_of_week, and time_since_signup (in hours)."""
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
    return df

def engineer_fraud_features(df):
    """Run all feature engineering steps for Fraud_Data.csv."""
    df = preprocess_datetime_columns(df)
    df = add_transaction_frequency(df)
    df = add_transaction_velocity(df)
    df = add_time_features(df)
    return df
