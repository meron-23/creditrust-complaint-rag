# src/preprocessing.py
import pandas as pd

def load_complaints(path: str) -> pd.DataFrame:
    """Load complaints dataset from CSV"""
    return pd.read_csv(path)

def clean_text(text: str) -> str:
    """Basic text cleaning - expand later as needed"""
    if not isinstance(text, str):
        return ""
    return text.strip().replace("\n", " ")

def preprocess_dataset(df: pd.DataFrame, text_col="cleaned_narrative") -> pd.DataFrame:
    df[text_col] = df[text_col].apply(clean_text)
    return df
