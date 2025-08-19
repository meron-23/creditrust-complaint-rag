import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss

from src.utils.exceptions import RetrievalError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ComplaintRetriever:
    def __init__(self, embedding_model: SentenceTransformer, index, metadata: List[Dict]):
        self.embedding_model = embedding_model
        self.index = index
        self.metadata = metadata
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query string"""
        return self.embedding_model.encode([query])[0]
    
    def retrieve_chunks(self, query: str, k: int = 5, 
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Retrieve relevant chunks with optional filtering"""
        try:
            # Semantic search
            query_vector = self.embed_query(query).astype('float32')
            
            # Search for results
            distances, indices = self.index.search(
                np.array([query_vector]), k * 2  # Get extra to allow filtering
            )
            
            # Apply filters and format results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.metadata) and idx >= 0:
                    metadata_item = self.metadata[idx]
                    
                    if self._passes_filters(metadata_item, filters):
                        results.append({
                            'text': metadata_item['text_chunk'],
                            'metadata': {k: v for k, v in metadata_item.items() if k != 'text_chunk'},
                            'score': float(distances[0][i])
                        })
                    
                    if len(results) >= k:
                        break
            
            logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")
            return results
            
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve chunks: {str(e)}")
    
    def _passes_filters(self, metadata: Dict, filters: Optional[Dict]) -> bool:
        """Check if metadata passes all filters"""
        if not filters:
            return True
        
        for key, value in filters.items():
            if key in metadata and metadata[key] != value:
                return False
        return True