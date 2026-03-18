import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def clean_data(df):
    df = df.dropna()

    # Drop irrelevant columns if needed
    if 'filename' in df.columns:
        df = df.drop(columns=['filename'])

    return df


def split_features_labels(df):
    X = df.drop('class', axis=1)
    y = df['class']
    return X, y

def encode_labels(y):
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder