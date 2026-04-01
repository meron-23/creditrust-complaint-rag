import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Core RAG settings
    CHUNK_SIZE: int = 350
    CHUNK_OVERLAP: int = 60
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_NAME: str = "models/gemini-3-flash-preview"
    VECTOR_STORE_PATH: str = "vector_store/credtrust_bi_index"
    DATA_PATH: str = "./data/filtered_complaints.csv"
    
    # CrediTrust specific settings
    MAX_GENERATION_LENGTH: int = 1000
    TEMPERATURE: float = 0.1
    TOP_K_RETRIEVAL: int = 5
    
    # CrediTrust products and markets - using default_factory for mutable defaults
    PRODUCTS: List[str] = field(default_factory=lambda: ["Credit Cards", "Personal Loans", "Buy Now, Pay Later (BNPL)", "Savings Accounts", "Money Transfers"])
    MARKETS: List[str] = field(default_factory=lambda: ["Kenya", "Uganda", "Tanzania", "Rwanda"])
    
    @classmethod
    def from_env(cls):
        return cls(
            CHUNK_SIZE=int(os.getenv('CHUNK_SIZE', 350)),
            CHUNK_OVERLAP=int(os.getenv('CHUNK_OVERLAP', 60)),
            EMBEDDING_MODEL_NAME=os.getenv('EMBEDDING_MODEL', "sentence-transformers/all-MiniLM-L6-v2"),
            LLM_MODEL_NAME=os.getenv('LLM_MODEL', "models/gemini-3-flash-preview"),
            VECTOR_STORE_PATH=os.getenv('VECTOR_STORE_PATH', "vector_store/credtrust_bi_index"),
            DATA_PATH=os.getenv('DATA_PATH', "./data/filtered_complaints.csv"),
            MAX_GENERATION_LENGTH=int(os.getenv('MAX_GENERATION_LENGTH', 1000)),
            TEMPERATURE=float(os.getenv('TEMPERATURE', 0.1)),
            TOP_K_RETRIEVAL=int(os.getenv('TOP_K_RETRIEVAL', 5))
        )