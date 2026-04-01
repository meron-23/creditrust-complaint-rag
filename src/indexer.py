import os
import pickle
import pandas as pd
from typing import Dict, List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from src.utils.exceptions import IndexingError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

from src.preprocessing import map_product_to_group
import random

class ComplaintIndexer:
    def __init__(self, config):
        self.config = config
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.index = None
        self.metadatas = []
    
    def build_index(self, df: pd.DataFrame, text_col: str = "cleaned_narrative"):
        """Build FAISS index from dataframe"""
        try:
            texts, metadatas = [], []
            markets = ["Kenya", "Uganda", "Tanzania", "Rwanda"]
            
            for i, row in df.iterrows():
                # Get the narrative text (use cleaned version if available)
                narrative = row.get(text_col, row.get('Consumer complaint narrative', ''))
                if not narrative or not isinstance(narrative, str):
                    continue
                    
                chunks = self.splitter.split_text(narrative)
                
                # Determine product group and simulated market
                orig_product = row.get('Product', 'Unknown')
                product_group = map_product_to_group(orig_product)
                
                # Use existing market column if it exists, otherwise simulate East African context
                market = row.get('market', random.choice(markets))
                
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append({
                        "complaint_id": row.get('Complaint ID', row.get('complaint_id', f'CT_{i}')),
                        "product": product_group,
                        "original_product": orig_product,
                        "market": market,
                        "date": row.get('Date received', row.get('date', 'Unknown')),
                        "severity": row.get('severity', 'Medium'),
                        "channel": row.get('Submitted via', row.get('channel', 'In-app')),
                        "text_chunk": chunk,
                    })
            logger.info(f"Generated {len(texts)} chunks from {len(df)} complaints")
            
            # Create embeddings
            logger.info("Generating embeddings...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            self._create_index(embeddings, metadatas)
            logger.info("Index built successfully")
            
        except Exception as e:
            raise IndexingError(f"Failed to build index: {str(e)}")
    
    def _create_index(self, embeddings: np.ndarray, metadatas: List[Dict]):
        """Create and populate FAISS index"""
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings.astype('float32'))
        self.metadatas = metadatas
    
    def save(self):
        """Save index and metadata to disk"""
        try:
            os.makedirs(os.path.dirname(self.config.VECTOR_STORE_PATH), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, self.config.VECTOR_STORE_PATH + ".index")
            
            # Save metadata
            with open(self.config.VECTOR_STORE_PATH + "_meta.pkl", "wb") as f:
                pickle.dump(self.metadatas, f)
            
            logger.info(f"Saved index to {self.config.VECTOR_STORE_PATH}.index")
            logger.info(f"Saved metadata to {self.config.VECTOR_STORE_PATH}_meta.pkl")
            
        except Exception as e:
            raise IndexingError(f"Failed to save index: {str(e)}")
    
    def load(self):
        """Load index and metadata from disk"""
        try:
            if not os.path.exists(self.config.VECTOR_STORE_PATH + ".index"):
                raise IndexingError("Index file not found")
            
            self.index = faiss.read_index(self.config.VECTOR_STORE_PATH + ".index")
            
            with open(self.config.VECTOR_STORE_PATH + "_meta.pkl", "rb") as f:
                self.metadatas = pickle.load(f)
            
            logger.info("Index loaded successfully")
            
        except Exception as e:
            raise IndexingError(f"Failed to load index: {str(e)}")