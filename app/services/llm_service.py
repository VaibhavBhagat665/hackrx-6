import google.generativeai as genai
from typing import List, Dict, Tuple
import re
from app.core.config import settings
from app.core.logging import logger

class LLMService:
    """advanced llm processing with chain of thought"""
    
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model)
        logger.info("llm service initialized")
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> Dict[str, any]:
        """generate answer with reasoning"""
        
        try:
            # prepare context
            context = self._format_context(context_chunks)
            
            # create prompt
            prompt = self._create_cot_prompt(query, context)
            
            # generate response
            response = self.model.generate_content(prompt)
            
            # parse response
            parsed = self._parse_response(response.text)
            
            return parsed
            
        except Exception as e:
            logger.error(f"llm generation failed: {str(e)}")
            return {
                'answer': "unable to process query",
                'confidence': 0.0,
                'reasoning': f"error: {str(e)}"
            }
    
    def _format_context(self, chunks: List[str]) -> str:
        """format context chunks for llm"""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"[CONTEXT {i}]:\n{chunk}\n")
        
        return '\n'.join(formatted)
    
    def _create_cot_prompt(self, query: str, context: str) -> str:
        """create chain of thought prompt"""
        
        prompt = f"""You are an expert document analyst. Analyze the provided context and answer the question using chain of thought reasoning.

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Identify key concepts in the question
2. Search through context for relevant information
3. Analyze evidence piece by piece
4. Synthesize a comprehensive answer
5. Provide confidence score (0.0-1.0)

RESPONSE FORMAT:
ANALYSIS: [your step-by-step reasoning]
EVIDENCE: [specific quotes and references]
ANSWER: [direct, comprehensive answer]
CONFIDENCE: [0.0-1.0 confidence score]

Begin your analysis:"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict[str, any]:
        """parse structured llm response"""
        
        try:
            # extract sections
            analysis_match = re.search(r'ANALYSIS:\s*(.*?)(?=EVIDENCE:|$)', response_text, re.DOTALL)
            evidence_match = re.search(r'EVIDENCE:\s*(.*?)(?=ANSWER:|$)', response_text, re.DOTALL)
            answer_match = re.search(r'ANSWER:\s*(.*?)(?=CONFIDENCE:|$)', response_text, re.DOTALL)
            confidence_match = re.search(r'CONFIDENCE:\s*([\d\.]+)', response_text)
            
            # extract content
            analysis = analysis_match.group(1).strip() if analysis_match else ""
            evidence = evidence_match.group(1).strip() if evidence_match else ""
            answer = answer_match.group(1).strip() if answer_match else response_text
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'answer': answer,
                'confidence': confidence,
                'reasoning': analysis,
                'evidence': evidence
            }
            
        except Exception as e:
            logger.error(f"response parsing failed: {str(e)}")
            return {
                'answer': response_text,
                'confidence': 0.5,
                'reasoning': "parsing failed",
                'evidence': ""
            }
