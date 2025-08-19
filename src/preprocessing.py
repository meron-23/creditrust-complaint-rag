import re
import pandas as pd
import os
from typing import List, Optional
from src.utils.exceptions import DataLoadingError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_complaints(path: str) -> pd.DataFrame:
    """Load complaints dataset from CSV with error handling"""
    try:
        if not os.path.exists(path):
            raise DataLoadingError(f"Data file not found: {path}")
        
        df = pd.read_csv(path)
        logger.info(f"Loaded dataset with {len(df)} rows from {path}")
        return df
    except Exception as e:
        raise DataLoadingError(f"Failed to load data from {path}: {str(e)}")

def clean_text(text: str) -> str:
    """Comprehensive text cleaning"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
    return text

def preprocess_dataset(df: pd.DataFrame, text_col: str = "cleaned_narrative") -> pd.DataFrame:
    """Enhanced preprocessing with validation"""
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in dataframe")
    
    original_size = len(df)
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.len() > 10]  # Remove very short texts
    
    logger.info(f"Preprocessing: {original_size} -> {len(df)} rows after cleaning")
    return df