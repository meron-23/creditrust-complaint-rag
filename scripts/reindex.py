import os
import pandas as pd
from src.config import Config
from src.preprocessing import load_complaints, preprocess_dataset
from src.indexer import ComplaintIndexer

def force_reindex():
    print("--- CrediTrust AI: Forced Re-indexing ---")
    config = Config.from_env()
    
    # Load and preprocess (Limit to 1000 rows for demo speed)
    print(f"Loading top 1000 rows from {config.DATA_PATH}...")
    df = load_complaints(config.DATA_PATH).head(1000)
    df_processed = preprocess_dataset(df)
    
    # Build index
    print(f"Building new index at {config.VECTOR_STORE_PATH}...")
    indexer = ComplaintIndexer(config)
    indexer.build_index(df_processed)
    indexer.save()
    
    print(f"Re-indexing complete. New index saved to {config.VECTOR_STORE_PATH}")

if __name__ == "__main__":
    force_reindex()
