# src/rag_pipeline.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List
from transformers import pipeline

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the persisted FAISS index and metadata
with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
index = faiss.read_index("vector_store/index.faiss")

# Load the LLM (you can swap this for LangChain or OpenAI if needed)
generator = pipeline("text-generation", model="gpt2", max_length=500)

def embed_query(question: str) -> np.ndarray:
    return embedding_model.encode([question])[0]

def retrieve_chunks(query: str, k: int = 5):
    query_vector = embed_query(query).astype("float32")
    distances, indices = index.search(np.array([query_vector]), k)
    
    chunks = []
    for idx in indices[0]:
        chunks.append(metadata[idx])
    return chunks

def build_prompt(context_chunks: List[str], question: str) -> str:
    context = "\n---\n".join(context_chunks)
    prompt = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""
    return prompt.strip()

def generate_answer(prompt: str) -> str:
    result = generator(prompt, do_sample=True, top_k=50, top_p=0.95)[0]
    return result['generated_text'].split("Answer:")[-1].strip()

def rag_pipeline(question: str, k: int = 5):
    chunks = retrieve_chunks(question, k)
    prompt = build_prompt(chunks, question)
    answer = generate_answer(prompt)
    return answer, chunks
