class RAGException(Exception):
    """Base exception for RAG pipeline"""
    pass

class IndexingError(RAGException):
    """Error during index building"""
    pass

class RetrievalError(RAGException):
    """Error during retrieval"""
    pass

class GenerationError(RAGException):
    """Error during answer generation"""
    pass

class DataLoadingError(RAGException):
    """Error during data loading"""
    pass