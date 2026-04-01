from dotenv import load_dotenv
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.indexer import ComplaintIndexer
from src.retriever import ComplaintRetriever
from src.generator import BusinessAnswerGenerator
from src.rag_pipeline import RAGPipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def verify():
    load_dotenv()
    config = Config.from_env()
    
    # Initialize components
    indexer = ComplaintIndexer(config)
    indexer.load() # Assuming index exists
    
    retriever = ComplaintRetriever(indexer.model, indexer.index, indexer.metadatas)
    generator = BusinessAnswerGenerator(config)
    pipeline = RAGPipeline(retriever, generator, config)
    
    question = "What are the top complaints about Credit Cards in Kenya?"
    print(f"\nProcessing query: '{question}'")
    
    # Run pipeline with NO explicit filters (should auto-extract)
    answer, chunks = pipeline.run(question, k=5, filters={})
    
    print("\n--- Pipeline Result ---")
    print(f"Answer snippet: {answer[:300]}...")
    print(f"\nRetrieved {len(chunks)} chunks.")
    
    valid_market = True
    for i, chunk in enumerate(chunks):
        market = chunk['metadata'].get('market', 'Unknown')
        product = chunk['metadata'].get('product', 'Unknown')
        print(f" Chunk {i+1} - Market: {market}, Product: {product}")
        if market != "Kenya":
            valid_market = False
            
    if valid_market:
        print("\n✅ Verification SUCCESS: All items are from Kenya.")
    else:
        print("\n❌ Verification FAILURE: Some items are not from Kenya.")

if __name__ == "__main__":
    verify()
