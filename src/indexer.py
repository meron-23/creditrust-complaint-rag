import os
import pickle
import pandas as pd
from typing import Dict, List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from src.utils.exceptions import IndexingError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

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
            
            for i, row in df.iterrows():
                chunks = self.splitter.split_text(row[text_col])
                
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append({
                        "complaint_id": row.get('complaint_id', f'CT_{i}'),
                        "product": row.get('product', 'Unknown'),
                        "market": row.get('market', 'Unknown'),
                        "date": row.get('date', 'Unknown'),
                        "severity": row.get('severity', 'Medium'),
                        "channel": row.get('channel', 'Unknown'),
                        "text_chunk": chunk,
                        "chunk_length": len(chunk)
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