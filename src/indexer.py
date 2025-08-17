# src/indexer.py
import os, pickle, faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

class ComplaintIndexer:
    def __init__(self, model_name, chunk_size, chunk_overlap, vector_store_path):
        self.model = SentenceTransformer(model_name)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vector_store_path = vector_store_path
        self.index = None
        self.metadatas = []

    def build_index(self, df, text_col="cleaned_narrative", product_col="Product", id_col="Complaint ID"):
        texts = []
        for i, row in df.iterrows():
            complaint_id = row[id_col] if id_col in row else i
            product = row[product_col]
            text = row[text_col]
            chunks = self.splitter.split_text(text)

            for chunk in chunks:
                texts.append(chunk)
                self.metadatas.append({
                    "complaint_id": complaint_id,
                    "product": product,
                    "text_chunk": chunk
                })

        embeddings = self.model.encode(texts, show_progress_bar=True)
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings)

    def save(self):
        os.makedirs("vector_store", exist_ok=True)
        faiss.write_index(self.index, self.vector_store_path + ".index")
        with open(self.vector_store_path + "_meta.pkl", "wb") as f:
            pickle.dump(self.metadatas, f)

    def load(self):
        self.index = faiss.read_index(self.vector_store_path + ".index")
        with open(self.vector_store_path + "_meta.pkl", "rb") as f:
            self.metadatas = pickle.load(f)
