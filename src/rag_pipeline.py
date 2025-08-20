from typing import Tuple, List, Dict
from src.utils.logger import setup_logger
from src.query_validator import QueryValidator

logger = setup_logger(__name__)

class RAGPipeline:
    def __init__(self, retriever, generator, config):
        self.retriever = retriever
        self.generator = generator
        self.validator = QueryValidator(config)  # ← Pass config to validator
        self.config = config
    
    def run(self, question: str, k: int = 5, filters: Dict = None) -> Tuple[str, List[Dict]]:
        """Run the complete RAG pipeline with query validation"""
        try:
            logger.info(f"Processing question: {question}")
            
            # Validate query first
            is_valid, validation_message = self.validator.validate_query(question)
            if not is_valid:
                return validation_message + self.validator.suggest_questions(), []
            
            # Retrieve relevant chunks
            chunks = self.retriever.retrieve_chunks(question, k, filters)
            
            if not chunks:
                return "I couldn't find any relevant information to answer your question. Please try rephrasing or ask about a different topic related to financial complaints.", []
            
            # Build prompt and generate answer
            prompt = self.generator.build_prompt(chunks, question)
            answer = self.generator.generate_answer(prompt)
            
            logger.info("RAG pipeline completed successfully")
            return answer, chunks
            
        except Exception as e:
            error_msg = f"RAG pipeline failed: {str(e)}"
            logger.error(error_msg)
            return "I'm sorry, I encountered an error processing your request. Please try again.", []
    
    def _analyze_complaint_patterns(self, chunks: List[Dict], question: str) -> List[Dict]:
        """Analyze complaint patterns for 'top' questions"""
        # Simple frequency analysis
        complaint_themes = {}
        
        for chunk in chunks:
            text = chunk['text'].lower()
            
            # Simple theme detection (you could make this more sophisticated)
            themes = self._detect_complaint_themes(text)
            for theme in themes:
                complaint_themes[theme] = complaint_themes.get(theme, 0) + 1
        
        # Sort by frequency
        sorted_themes = sorted(complaint_themes.items(), key=lambda x: x[1], reverse=True)
        
        # Add theme information to metadata for the generator
        for chunk in chunks:
            chunk['metadata']['themes'] = self._detect_complaint_themes(chunk['text'].lower())
        
        logger.info(f"Detected complaint themes: {sorted_themes[:5]}")
        return chunks
    
    def _detect_complaint_themes(self, text: str) -> List[str]:
        """Detect common complaint themes in text"""
        themes = []
        text_lower = text.lower()
        
        # Credit card specific themes
        if any(term in text_lower for term in ['fee', 'charge', 'annual fee', 'hidden fee']):
            themes.append('hidden_fees')
        if any(term in text_lower for term in ['interest', 'apr', 'rate increase']):
            themes.append('interest_rates')
        if any(term in text_lower for term in ['fraud', 'unauthorized', 'stolen', 'identity theft']):
            themes.append('fraud')
        if any(term in text_lower for term in ['service', 'customer service', 'representative', 'wait']):
            themes.append('poor_service')
        if any(term in text_lower for term in ['credit limit', 'limit decrease', 'credit line']):
            themes.append('credit_limit_issues')
        if any(term in text_lower for term in ['payment', 'late payment', 'due date']):
            themes.append('payment_issues')
        if any(term in text_lower for term in ['dispute', 'chargeback', 'billing error']):
            themes.append('dispute_resolution')
        if any(term in text_lower for term in ['application', 'denied', 'approval', 'credit score']):
            themes.append('application_issues')
        
        return themes