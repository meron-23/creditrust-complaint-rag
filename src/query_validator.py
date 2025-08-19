import re
from typing import Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class QueryValidator:
    def __init__(self, config):
        self.config = config
        self.casual_patterns = [
            r'^(hi|hello|hey|howdy|greetings|sup|yo|what\'s up|wassup)',
            r'^(thanks|thank you|thx|ty|cheers)',
            r'^(bye|goodbye|see ya|cya|later)',
            r'^(ok|okay|k|alright|sure|fine)',
            r'^(\?|\.|!|,|;|:)',
            r'^.{1,3}$',
        ]
        
        self.business_question_patterns = [
            r'^(top|most common|frequent|common|biggest|emerging) (issue|problem|complaint|concern)',
            r'^(what are|list|identify|analyze) (the|some) (common|frequent|top)',
            r'^(trend|pattern|theme) (in|with|for)',
            r'^(credit card|loan|bnpl|buy now pay later|savings|money transfer)',
            r'^(app|mobile|digital|platform) (issue|problem|bug|error)',
            r'^(kenya|uganda|tanzania|rwanda|east africa)',
            r'^(customer satisfaction|user experience|cx|support)',
            r'^(regulatory|compliance|cbk|central bank)',
        ]
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """Validate if query is appropriate for business analysis"""
        query = query.strip().lower()
        
        if not query or len(query) < 2:
            return False, "Please provide a specific question about customer complaints."
        
        # Check for casual conversation
        for pattern in self.casual_patterns:
            if re.match(pattern, query, re.IGNORECASE):
                return False, "I'm here to help analyze customer complaints for business insights. Please ask about specific products or issues."
        
        # Check if it's a business-relevant question
        is_business_question = any(re.search(pattern, query, re.IGNORECASE) for pattern in self.business_question_patterns)
        
        if not is_business_question and len(query.split()) < 4:
            return False, "This doesn't appear to be a business analysis question. I specialize in customer complaint insights for CrediTrust products."
        
        return True, "Valid business query"
    
    def suggest_questions(self) -> str:
        """Provide business-relevant question examples"""
        return """
🎯 **Business Analysis Questions for CrediTrust:**

**Product Insights:**
- What are the top complaints about BNPL in Kenya?
- What emerging issues are we seeing with mobile money transfers?
- Analyze complaint trends for credit cards in Uganda

**Operational Issues:**
- What are the most common app functionality complaints?
- What payment processing issues are customers reporting?
- Analyze customer service complaint patterns

**Geographic Analysis:**
- Compare complaint themes between Kenya and Tanzania
- What are the unique issues in the Rwandan market?
- Regional trends in personal loan complaints

**Strategic Questions:**
- What regulatory concerns are emerging from complaints?
- What features are customers requesting most frequently?
- Identify potential fraud patterns from complaints
"""