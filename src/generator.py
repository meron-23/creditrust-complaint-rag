# src/generator.py
from transformers import pipeline
from typing import List

class AnswerGenerator:
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline("text-generation", model=model_name, max_length=500)

    def build_prompt(self, context_chunks: List[str], question: str) -> str:
        context = "\n---\n".join(context_chunks)
        return f"""
        You are a financial analyst assistant for CrediTrust.
        Use the following complaint excerpts to answer questions.
        If context is insufficient, say so.

        Context:
        {context}

        Question: {question}
        Answer:
        """.strip()

    def generate_answer(self, prompt: str) -> str:
        result = self.generator(prompt, do_sample=True, top_k=50, top_p=0.95)[0]
        return result['generated_text'].split("Answer:")[-1].strip()
