# app.py
import streamlit as st
import os
from src import config
from src.preprocessing import load_complaints, preprocess_dataset
from src.indexer import ComplaintIndexer
from src.retriever import ComplaintRetriever
from src.generator import AnswerGenerator
from src.rag_pipeline import RAGPipeline

# ---------------------------
# Setup
# ---------------------------
st.set_page_config(page_title="Intelligent Complaint Analysis", layout="wide")

@st.cache_resource
def load_pipeline():
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
        indexer.build_index(df)
        indexer.save()
    else:
        indexer.load()

    # 3. Initialize pipeline
    retriever = ComplaintRetriever(indexer.model, indexer.index, indexer.metadatas)
    generator = AnswerGenerator(model_name=config.LLM_MODEL_NAME)
    return RAGPipeline(retriever, generator)

rag_pipeline = load_pipeline()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("💡 Intelligent Complaint Analysis (Financial Services)")

st.markdown(
    "Ask questions about financial complaints. The system retrieves complaint excerpts and generates an answer using RAG."
)

query = st.text_input("🔍 Enter your question:", placeholder="e.g. What are common issues with credit cards?")

if query:
    with st.spinner("Analyzing complaints..."):
        answer, chunks = rag_pipeline.run(query, k=3)

    st.subheader("📌 Answer")
    st.write(answer)

    with st.expander("🔎 Context Used"):
        for i, c in enumerate(chunks):
            st.markdown(f"**Chunk {i+1}:**\n{c}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, FAISS, and Transformers")
