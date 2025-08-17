# src/rag_pipeline.py
from src.retriever import ComplaintRetriever
from src.generator import AnswerGenerator

class RAGPipeline:
    def __init__(self, retriever: ComplaintRetriever, generator: AnswerGenerator):
        self.retriever = retriever
        self.generator = generator

    def run(self, question: str, k: int = 5):
        chunks = self.retriever.retrieve_chunks(question, k)
        prompt = self.generator.build_prompt(chunks, question)
        answer = self.generator.generate_answer(prompt)
        return answer, chunks
