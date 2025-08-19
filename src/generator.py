from transformers import pipeline, AutoTokenizer
from typing import List, Dict
import torch
import re
from datetime import datetime

from src.utils.exceptions import GenerationError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class BusinessAnswerGenerator:
    def __init__(self, config):
        self.config = config
        try:
            self.generator = pipeline(
                "text2text-generation",
                model=config.LLM_MODEL_NAME,
                max_length=config.MAX_GENERATION_LENGTH,
                truncation=True,
                temperature=config.TEMPERATURE,
                device=0 if torch.cuda.is_available() else -1
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
            logger.info(f"Loaded business generator model: {config.LLM_MODEL_NAME}")
            
        except Exception as e:
            raise GenerationError(f"Failed to initialize generator: {str(e)}")
    
    def build_prompt(self, context_chunks: List[Dict], question: str) -> str:
        """Build business-focused prompt for CrediTrust analysis"""
        context_str = "\n\n".join([
            f"COMPLAINT {i+1} (Product: {chunk['metadata'].get('product', 'N/A')}, "
            f"Market: {chunk['metadata'].get('market', 'N/A')}, "
            f"Date: {chunk['metadata'].get('date', 'N/A')}):\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""**CREDITRUST FINANCIAL - BUSINESS INTELLIGENCE ANALYSIS**
Date: {datetime.now().strftime('%Y-%m-%d')}
Analyst: AI Complaint Insights Tool

**BUSINESS QUESTION:**
{question}

**RELEVANT CUSTOMER COMPLAINTS ({len(context_chunks)} excerpts):**
{context_str}

**ANALYSIS FRAMEWORK:**
1. QUANTITATIVE INSIGHTS: Count frequency of specific issues mentioned
2. QUALITATIVE THEMES: Identify patterns and emerging trends  
3. PRODUCT IMPACT: Relate issues to specific CrediTrust products
4. GEOGRAPHIC CONTEXT: Note any regional patterns
5. BUSINESS IMPLICATIONS: Suggest potential actions or investigations
6. CONFIDENCE LEVEL: Note data limitations and evidence strength

**REQUIRED OUTPUT FORMAT:**
- Executive Summary: 2-3 sentence overview of key findings
- Key Findings: Bulleted list of quantified insights with complaint evidence
- Geographic Patterns: Regional variations if apparent
- Product Impact: Which products are most affected
- Recommended Actions: 2-3 specific next steps for investigation
- Data Limitations: Scope and confidence notes

**ANALYSIS:**"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """Generate business-ready analysis"""
        try:
            result = self.generator(
                prompt,
                max_length=600,
                num_beams=4,
                early_stopping=True,
                temperature=0.2,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
            
            generated_text = result[0]['generated_text']
            
            # Extract analysis part
            if "ANALYSIS:" in generated_text:
                analysis = generated_text.split("ANALYSIS:")[-1].strip()
            else:
                analysis = generated_text.strip()
            
            # Format for business readability
            analysis = self._format_business_output(analysis)
            
            logger.info("Business analysis generated successfully")
            return analysis
            
        except Exception as e:
            error_msg = f"Failed to generate business analysis: {str(e)}"
            logger.error(error_msg)
            return "**SYSTEM ERROR**\nI encountered a technical issue generating the analysis. Please try again with a different question or contact technical support."
    
    def _format_business_output(self, analysis: str) -> str:
        """Format the analysis for business readability"""
        analysis = re.sub(r'\n+', '\n', analysis)
        analysis = analysis.replace('•', '  • ')
        analysis = analysis.replace('-', '  - ')
        return analysis