from google import genai
from typing import List, Dict
import os
import re
from datetime import datetime

from src.utils.exceptions import GenerationError
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class BusinessAnswerGenerator:
    def __init__(self, config):
        self.config = config
        self.model_name = config.LLM_MODEL_NAME
        try:
            # Configure Gemini via the new Client
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY not found in environment. Please ensure it is set.")
            
            self.client = genai.Client(api_key=api_key)
            logger.info(f"Initialized Gemini generator with model: {self.model_name}")
            
        except Exception as e:
            raise GenerationError(f"Failed to initialize Gemini generator: {str(e)}")
    
    def build_prompt(self, context_chunks: List[Dict], question: str) -> str:
        """Build a professional, business-focused prompt for Gemini analysis"""
        context_str = "\n\n".join([
            f"SOURCE COMPLAINT {i+1}\n"
            f"Product: {chunk['metadata'].get('product', 'N/A')}\n"
            f"Region: {chunk['metadata'].get('market', 'N/A')}\n"
            f"Date: {chunk['metadata'].get('date', 'N/A')}\n"
            f"Narrative: {chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        prompt = f"""You are a Senior Financial Analyst at CrediTrust Financial, specializing in customer experience and operational risk for the East African market.

Your objective is to provide a synthesis of the following customer complaint excerpts to answer a specific business question. 

### BUSINESS QUESTION
{question}

### SOURCE DATA EXCERPTS
{context_str}

### ANALYSIS INSTRUCTIONS
1. **Be Professional & Objective**: Use a formal business tone. Avoid all emojis.
2. **Synthesize, Don't List**: Identify common themes across multiple complaints rather than just summarizing them individually.
3. **East African Context**: Where applicable, note regional patterns (Kenya, Uganda, Tanzania, Rwanda).
4. **Actionable Insights**: Provide specific, data-backed recommendations for product or support teams.
5. **Groundedness**: Only use information present in the provided excerpts. If the answer is not in the context, state it clearly.

### OUTPUT STRUCTURE
- **Executive Summary**: A concise (3-4 sentence) high-level overview.
- **Critical Issues Identified**: A prioritized list of recurring pain points.
- **Regional Considerations**: Market-specific patterns if observed.
- **Operational Recommendations**: Strategic next steps for the business.

Your analysis:"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """Generate analysis using the Gemini API"""
        try:
            logger.info("Sending prompt to Gemini...")
            
            # Using the new google-genai SDK method
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            
            if not response.text:
                raise GenerationError("Gemini returned an empty response")
            
            analysis = response.text.strip()
            
            # Clean up any residual emojis or informal formatting if the LLM hallucinated them
            analysis = self._sanitize_business_output(analysis)
            
            logger.info("Business analysis generated successfully via Gemini")
            return analysis
            
        except Exception as e:
            error_msg = f"Failed to generate analysis via Gemini: {str(e)}"
            logger.error(error_msg)
            return "TECHNICAL ERROR: I encountered an issue connecting to the analysis engine. Please verify your API configuration and try again."
    
    def _sanitize_business_output(self, analysis: str) -> str:
        """Ensure the output is professional and emoji-free"""
        # Remove common emojis if they slipped through
        analysis = re.sub(r'[^\x00-\x7F]+', '', analysis) 
        # Ensure clean formatting
        analysis = analysis.replace('•', '-')
        return analysis.strip()