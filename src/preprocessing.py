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
    """Comprehensive text cleaning for CrediTrust analytics"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove boilerplate often found in CFPB data
    text = re.sub(r"i am writing to file a complaint regarding", "", text)
    text = re.sub(r"i am writing to complain about", "", text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-()]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def map_product_to_group(product_name: str) -> str:
    """Map raw CFPB products to CrediTrust standard categories"""
    pn = product_name.lower()
    if "credit card" in pn: return "Credit Cards"
    if "loan" in pn: return "Personal Loans"
    if "bnpl" in pn: return "Buy Now, Pay Later (BNPL)"
    if "checking" in pn or "savings" in pn: return "Savings Accounts"
    if "money transfer" in pn or "money service" in pn: return "Money Transfers"
    return "Other"

def preprocess_dataset(df: pd.DataFrame, text_col: str = "Consumer complaint narrative") -> pd.DataFrame:
    """Enhanced preprocessing for RAG pipeline"""
    if text_col not in df.columns:
        # Check if it was already cleaned and renamed
        if "cleaned_narrative" in df.columns:
            return df
        raise ValueError(f"Text column '{text_col}' not found in dataframe")
    
    original_size = len(df)
    
    # 1. Filter empty narratives
    df = df.dropna(subset=[text_col])
    
    # 2. Add product grouping if not present
    if 'product_group' not in df.columns:
        df['product_group'] = df['Product'].apply(map_product_to_group)
    
    # 3. Clean narratives
    df['cleaned_narrative'] = df[text_col].apply(clean_text)
    
    # 4. Remove very short texts
    df = df[df['cleaned_narrative'].str.len() > 10]
    
    logger.info(f"Preprocessing: {original_size} -> {len(df)} rows after cleaning")
    return df