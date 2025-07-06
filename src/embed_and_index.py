import pandas as pd
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle

# Config
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store/faiss_index"

# 1. Load cleaned complaint dataset
df = pd.read_csv("data/filtered_complaints.csv")

# 2. Initialize chunker
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# 3. Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 4. Prepare containers
texts = []
metadatas = []
index_id = 0

print("Chunking and embedding complaints...")

# 5. Chunk and embed
for i, row in tqdm(df.iterrows(), total=len(df)):
    complaint_id = row['Complaint ID'] if 'Complaint ID' in row else i
    product = row['Product']
    text = row['cleaned_narrative']

    chunks = splitter.split_text(text)

    for chunk in chunks:
        texts.append(chunk)
        metadatas.append({
            "complaint_id": complaint_id,
            "product": product,
            "text_chunk": chunk
        })

# 6. Create vector embeddings
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# 7. Initialize FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)

# 8. Save FAISS index and metadata
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, os.path.join(VECTOR_STORE_PATH + ".index"))

with open(os.path.join(VECTOR_STORE_PATH + "_meta.pkl"), "wb") as f:
    pickle.dump(metadatas, f)

print(f"Saved vector index to: {VECTOR_STORE_PATH}.index")
print(f"Saved metadata to: {VECTOR_STORE_PATH}_meta.pkl")
