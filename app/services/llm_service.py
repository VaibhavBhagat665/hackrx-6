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
            
            # format the final answer for HackRx requirements
            parsed['answer'] = self.format_answer(parsed['answer'])
            
            return parsed
            
        except Exception as e:
            logger.error(f"llm generation failed: {str(e)}")
            return {
                'answer': "Unable to process query due to technical error.",
                'confidence': 0.0,
                'reasoning': f"error: {str(e)}"
            }
    
    def format_answer(self, raw_answer: str) -> str:
        """
        Format answer to meet HackRx requirements:
        - Single line only (no line breaks)
        - No markdown or special formatting
        - Concise and grammatically complete
        - Plain English only
        """
        if not raw_answer or not raw_answer.strip():
            return "No relevant information found in the document."
        
        # Remove markdown formatting
        formatted = raw_answer
        
        # Remove bold/italic markers
        formatted = re.sub(r'\*\*([^*]+)\*\*', r'\1', formatted)  # **bold**
        formatted = re.sub(r'\*([^*]+)\*', r'\1', formatted)      # *italic*
        formatted = re.sub(r'__([^_]+)__', r'\1', formatted)      # __bold__
        formatted = re.sub(r'_([^_]+)_', r'\1', formatted)        # _italic_
        
        # Remove headers and bullet points
        formatted = re.sub(r'^#+\s*', '', formatted, flags=re.MULTILINE)  # # headers
        formatted = re.sub(r'^\s*[-*+]\s*', '', formatted, flags=re.MULTILINE)  # bullet points
        formatted = re.sub(r'^\s*\d+\.\s*', '', formatted, flags=re.MULTILINE)  # numbered lists
        
        # Remove code blocks and inline code
        formatted = re.sub(r'```[^`]*```', '', formatted, flags=re.DOTALL)  # code blocks
        formatted = re.sub(r'`([^`]+)`', r'\1', formatted)  # inline code
        
        # Remove links but keep text
        formatted = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', formatted)  # [text](url)
        
        # Replace line breaks and multiple spaces with single spaces
        formatted = re.sub(r'\n+', ' ', formatted)
        formatted = re.sub(r'\s+', ' ', formatted)
        
        # Remove leading/trailing whitespace
        formatted = formatted.strip()
        
        # Remove quotes if the entire answer is wrapped in them
        if (formatted.startswith('"') and formatted.endswith('"')) or \
           (formatted.startswith("'") and formatted.endswith("'")):
            formatted = formatted[1:-1].strip()
        
        # Ensure it ends with proper punctuation
        if formatted and not re.search(r'[.!?]$', formatted):
            formatted += '.'
        
        # Truncate if too long while preserving sentence structure
        if len(formatted) > 200:
            # Find the last complete sentence within reasonable length
            sentences = re.split(r'[.!?]+', formatted)
            result = ""
            for sentence in sentences:
                if len(result + sentence.strip() + '.') <= 200:
                    if result:
                        result += ' '
                    result += sentence.strip()
                else:
                    break
            
            if result:
                if not re.search(r'[.!?]$', result):
                    result += '.'
                formatted = result
            else:
                # If no complete sentence fits, truncate and add ellipsis
                formatted = formatted[:197] + '...'
        
        # Final cleanup - ensure no empty result
        if not formatted or formatted.isspace():
            return "Information not available in the provided document."
        
        return formatted
    
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

IMPORTANT: Your ANSWER section must be a direct, concise response in plain English. Avoid bullet points, excessive formatting, or lengthy explanations. Focus on providing the specific information requested.

RESPONSE FORMAT:
ANALYSIS: [your step-by-step reasoning]
EVIDENCE: [specific quotes and references]
ANSWER: [direct, comprehensive answer in plain English]
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