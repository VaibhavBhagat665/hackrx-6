import httpx
from typing import List, Dict, Any
import re
import json
import asyncio
import ssl
import threading
from app.core.config import settings
from app.core.logging import logger
import time

class LLMService:
    """Ultra-optimized LLM processing for sub-30 second responses"""
    
    def __init__(self):
        """Initialize the OpenRouter client for Mistral Codestral 2508"""
        self.config = settings.get_llm_config()
        self.api_key = self.config["api_key"]
        self.model = self.config["model"]
        self.base_url = self.config["base_url"]
        self.timeout = 20  # Aggressive timeout for speed
        self.http_referer = self.config["http_referer"]
        self.x_title = self.config["x_title"]
        
        # Verify API key is configured
        if not self.api_key:
            logger.error("OpenRouter API key not configured")
            raise ValueError("OpenRouter API key is required")
        
        # Thread-local storage for HTTP clients (for thread safety)
        self._local = threading.local()
        
        logger.info(f"LLM service initialized with model: {self.model}")
        logger.info(f"Using provider: {self.config['provider']}")
    
    def _get_http_client(self):
        """Get thread-local HTTP client for thread safety"""
        if not hasattr(self._local, 'client'):
            self._local.client = httpx.Client(
                timeout=httpx.Timeout(15.0),  # Reduced timeout
                verify=False,  # Disable SSL verification for Cloud Run
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": self.http_referer,
                    "X-Title": self.x_title,
                    "User-Agent": "CloudRun-DocumentQuery/1.0"
                }
            )
        return self._local.client
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Ultra-fast answer generation optimized for speed
        
        Args:
            query: The user's question
            context_chunks: List of relevant document chunks
            
        Returns:
            Formatted single-line answer string
        """
        try:
            # Prepare context from chunks with aggressive truncation
            context = self._format_context_ultra_fast(context_chunks)
            
            # Create optimized prompt for maximum speed
            prompt = self._create_ultra_fast_prompt(query, context)
            
            # Call OpenRouter API with aggressive optimizations
            response = self._call_openrouter_api_fast(prompt)
            
            # Format the answer for HackRx requirements
            formatted_answer = self.format_answer_ultra_fast(response)
            
            logger.info(f"Generated answer for query: {query[:30]}...")
            return formatted_answer
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            # Return immediate fallback
            return "Information not available in the provided document."
    
    def _call_openrouter_api_fast(self, prompt: str) -> str:
        """
        Ultra-fast API call to OpenRouter with minimal retries
        
        Args:
            prompt: The formatted prompt to send
            
        Returns:
            The response text from the model
        """
        # Ultra-optimized payload for maximum speed
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # Minimum for fastest, most deterministic responses
            "max_tokens": 80,    # Reduced for speed
            "top_p": 0.9,
            "stream": False
        }
        
        # Single attempt with aggressive timeout
        try:
            client = self._get_http_client()
            
            logger.info("Making ultra-fast API call to OpenRouter")
            start_time = time.time()
            
            response = client.post(
                self.base_url,
                json=payload
            )
            
            api_time = time.time() - start_time
            logger.info(f"API response received in {api_time:.2f}s with status: {response.status_code}")
            
            response.raise_for_status()
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                logger.info("API call successful")
                return content
            else:
                raise ValueError("Invalid response format from OpenRouter API")
                
        except httpx.TimeoutException as e:
            logger.warning(f"Request timeout: {str(e)}")
            return "Request processing timed out."
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code}")
            if e.response.status_code == 429:
                return "Service temporarily busy, please try again."
            elif e.response.status_code >= 500:
                return "Service temporarily unavailable."
            else:
                return "Unable to process request at this time."
                
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return "Technical error occurred during processing."
    
    def format_answer_ultra_fast(self, raw_answer: str) -> str:
        """
        Ultra-fast answer formatting optimized for maximum speed
        
        Args:
            raw_answer: Raw answer from the model
            
        Returns:
            Single-line formatted answer
        """
        if not raw_answer or not raw_answer.strip():
            return "No relevant information found in the document."
        
        # Minimal cleanup for speed
        formatted = raw_answer.strip()
        
        # Single-pass cleanup with compiled regex for speed
        formatted = re.sub(r'[\n\r]+', ' ', formatted)  # Replace line breaks
        formatted = re.sub(r'\s+', ' ', formatted)      # Normalize spaces
        formatted = re.sub(r'[*#`]+', '', formatted)    # Remove markdown
        formatted = formatted.strip()
        
        # Remove wrapping quotes quickly
        if len(formatted) >= 2 and formatted[0] == '"' and formatted[-1] == '"':
            formatted = formatted[1:-1].strip()
        
        # Quick prefix removal
        if formatted.lower().startswith(('answer:', 'the answer is:', 'based on')):
            colon_pos = formatted.find(':')
            if colon_pos != -1:
                formatted = formatted[colon_pos + 1:].strip()
        
        # Ensure proper ending
        if formatted and not formatted[-1] in '.!?':
            formatted += '.'
        
        # Quick truncation if too long
        if len(formatted) > 200:
            formatted = formatted[:197] + '...'
        
        # Final validation
        if not formatted or formatted.isspace():
            return "Information not available in the provided document."
        
        return formatted
    
    def _format_context_ultra_fast(self, chunks: List[str]) -> str:
        """
        Ultra-fast context formatting for maximum speed
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "[NO CONTEXT]"
        
        # Take only top 2 chunks for speed, limit total size aggressively
        context_parts = []
        total_chars = 0
        max_context_chars = 1500  # Reduced for faster processing
        
        for i, chunk in enumerate(chunks[:2], 1):  # Only top 2 for maximum speed
            clean_chunk = chunk.strip()
            if clean_chunk and total_chars + len(clean_chunk) < max_context_chars:
                # Simple format for speed
                context_parts.append(f"[{i}] {clean_chunk}")
                total_chars += len(clean_chunk)
            else:
                break
        
        return '\n'.join(context_parts) if context_parts else "[NO VALID CONTEXT]"
    
    def _create_ultra_fast_prompt(self, query: str, context: str) -> str:
        """
        Create ultra-optimized prompt for maximum speed
        
        Args:
            query: User's question
            context: Formatted context chunks
            
        Returns:
            Ultra-optimized prompt for the model
        """
        # Minimal prompt for fastest processing
        prompt = f"""Context: {context}

Question: {query}

Answer directly in one sentence:"""
        
        return prompt
    
    def __del__(self):
        """Cleanup HTTP client when service is destroyed"""
        try:
            if hasattr(self._local, 'client'):
                self._local.client.close()
        except:
            pass