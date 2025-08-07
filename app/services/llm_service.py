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
        self.timeout = 25
        self.http_referer = self.config["http_referer"]
        self.x_title = self.config["x_title"]
        
        if not self.api_key:
            logger.error("OpenRouter API key not configured")
            raise ValueError("OpenRouter API key is required")
        
        self._local = threading.local()
        
        logger.info(f"LLM service initialized with model: {self.model}")
        logger.info(f"Using provider: {self.config['provider']}")
    
    def _get_http_client(self):
        """Get thread-local HTTP client for thread safety"""
        if not hasattr(self._local, 'client'):
            self._local.client = httpx.Client(
                timeout=httpx.Timeout(20.0),
                verify=False,
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
            context = self._format_context_ultra_fast(context_chunks)
            prompt = self._create_ultra_fast_prompt(query, context)
            response = self._call_openrouter_api_fast(prompt)
            formatted_answer = self.format_answer_ultra_fast(response)
            
            logger.info(f"Generated answer for query: {query[:30]}...")
            return formatted_answer
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return "Information not available in the provided document."
    
    def _call_openrouter_api_fast(self, prompt: str) -> str:
        """
        Ultra-fast API call to OpenRouter with minimal retries
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.temperature,
            "max_tokens": settings.max_tokens, # Use configured max_tokens
            "top_p": settings.top_p,
            "stream": False
        }
        
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
        """
        if not raw_answer or not raw_answer.strip():
            return "No relevant information found in the document."
        
        formatted = raw_answer.strip()
        
        formatted = re.sub(r'[\n\r]+', ' ', formatted)
        formatted = re.sub(r'\s+', ' ', formatted)
        formatted = re.sub(r'[*#`]+', '', formatted)
        formatted = formatted.strip()
        
        if len(formatted) >= 2 and formatted[0] == '"' and formatted[-1] == '"':
            formatted = formatted[1:-1].strip()
        
        if formatted.lower().startswith(('answer:', 'the answer is:', 'based on')):
            colon_pos = formatted.find(':')
            if colon_pos != -1:
                formatted = formatted[colon_pos + 1:].strip()
        
        if formatted and not formatted[-1] in '.!?':
            formatted += '.'
        
        # No hardcoded truncation
        
        if not formatted or formatted.isspace():
            return "Information not available in the provided document."
        
        return formatted
    
    def _format_context_ultra_fast(self, chunks: List[str]) -> str:
        """
        Ultra-fast context formatting for maximum speed
        """
        if not chunks:
            return "[NO CONTEXT]"
        
        context_parts = []
        total_chars = 0
        max_context_chars = 8000
        
        for i, chunk in enumerate(chunks[:settings.max_context_chunks], 1):
            clean_chunk = chunk.strip()
            if clean_chunk and total_chars + len(clean_chunk) < max_context_chars:
                context_parts.append(f"[{i}] {clean_chunk}")
                total_chars += len(clean_chunk)
            else:
                break
        
        return '\n'.join(context_parts) if context_parts else "[NO VALID CONTEXT]"
    
    def _create_ultra_fast_prompt(self, query: str, context: str) -> str:
        """
        Create ultra-optimized prompt for maximum speed
        """
        prompt = f"""You are an AI assistant designed to answer questions based *only* on the provided context.
If the context does not contain the information needed to answer the question, state that the information is not available in the document.

Context:
---
{context}
---

Question: {query}

Provide a complete, professional, and comprehensive answer based solely on the context. Do not use external knowledge."""
        
        return prompt
    
    def __del__(self):
        """Cleanup HTTP client when service is destroyed"""
        try:
            if hasattr(self._local, 'client'):
                self._local.client.close()
        except:
            pass