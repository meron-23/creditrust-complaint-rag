# main.py
import os
from src import config
from src.preprocessing import load_complaints, preprocess_dataset
from src.indexer import ComplaintIndexer
from src.retriever import ComplaintRetriever
from src.generator import AnswerGenerator
from src.rag_pipeline import RAGPipeline

def main():
    # 1. Load data
    df = load_complaints(config.DATA_PATH)
    df = preprocess_dataset(df)

    # 2. Build or load index
    indexer = ComplaintIndexer(
        model_name=config.EMBEDDING_MODEL_NAME,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        vector_store_path=config.VECTOR_STORE_PATH
    )

    if not os.path.exists(config.VECTOR_STORE_PATH + ".index"):
        print("Building index...")
        indexer.build_index(df)
        indexer.save()
    else:
        print("Loading existing index...")
        indexer.load()

    # 3. Initialize retriever + generator + pipeline
    retriever = ComplaintRetriever(indexer.model, indexer.index, indexer.metadatas)
    generator = AnswerGenerator(model_name=config.LLM_MODEL_NAME)
    rag = RAGPipeline(retriever, generator)

    # 4. Demo
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer, chunks = rag.run(query, k=3)
        print("\nAnswer:", answer)
        print("\nContext Used:")
        for c in chunks:
            print("-", c[:200], "...\n")

if __name__ == "__main__":
    main()
