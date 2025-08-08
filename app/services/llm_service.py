import httpx
from typing import List, Dict, Any, Optional
import re
import json
import threading
import time
import hashlib
import random
import asyncio
from datetime import datetime, timedelta
from app.core.config import settings
from app.core.logging import logger

class RateLimitManager:
    """Manages rate limiting across multiple providers"""
    
    def __init__(self):
        self.request_counts = {}
        self.reset_times = {}
        self.lock = threading.Lock()
    
    def can_make_request(self, provider: str, requests_per_minute: int = 60) -> bool:
        """Check if we can make a request to the provider"""
        with self.lock:
            now = datetime.now()
            
            # Reset counter if minute has passed
            if provider in self.reset_times:
                if now >= self.reset_times[provider]:
                    self.request_counts[provider] = 0
                    self.reset_times[provider] = now + timedelta(minutes=1)
            else:
                self.request_counts[provider] = 0
                self.reset_times[provider] = now + timedelta(minutes=1)
            
            # Check if under limit
            current_count = self.request_counts.get(provider, 0)
            if current_count < requests_per_minute:
                self.request_counts[provider] = current_count + 1
                return True
            
            return False
    
    def get_wait_time(self, provider: str) -> float:
        """Get time to wait before next request"""
        with self.lock:
            if provider in self.reset_times:
                wait_time = (self.reset_times[provider] - datetime.now()).total_seconds()
                return max(0, wait_time)
            return 0

class LLMService:
    """Enhanced LLM service optimized for precise, concise answers"""
    
    def __init__(self):
        """Initialize the LLM client with multi-provider support"""
        self.config = settings.get_llm_config()
        self.provider = self.config.get("provider", "gemini")
        self.api_key = self.config["api_key"]
        self.model = self.config["model"]
        self.base_url = self.config["base_url"]
        self.timeout = self.config.get("timeout", 45)
        
        # Rate limiting configuration
        self.rate_limit_manager = RateLimitManager()
        self.max_retries = getattr(settings, 'max_retries', 3)  # Reduced for efficiency
        self.base_delay = getattr(settings, 'base_delay', 1.0)
        
        # Provider-specific rate limits
        self.rate_limits = {
            'gemini': getattr(settings, 'gemini_requests_per_minute', 60),
            'openai': getattr(settings, 'openai_requests_per_minute', 60),
            'groq': getattr(settings, 'groq_requests_per_minute', 30),
            'openrouter': getattr(settings, 'openrouter_requests_per_minute', 20)
        }
        
        # Response cache
        self._response_cache = {}
        self._cache_lock = threading.Lock()
        
        # Request tracking
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'rate_limited_requests': 0,
            'failed_requests': 0
        }
        self.stats_lock = threading.Lock()
        
        if not self.api_key:
            logger.error(f"{self.provider.upper()} API key not configured")
            raise ValueError(f"{self.provider.upper()} API key is required")
        
        self._local = threading.local()
        logger.info(f"LLM service initialized with {self.provider}: {self.model}")
    
    def _get_http_client(self):
        """Get thread-local HTTP client"""
        if not hasattr(self._local, 'client'):
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "DocumentQuerySystem/2.0"
            }
            
            # Provider-specific headers
            if self.provider == "gemini":
                # Gemini uses API key in URL, not header
                pass
            elif self.provider in ["openai", "groq"]:
                headers["Authorization"] = f"Bearer {self.api_key}"
            elif self.provider == "openrouter":
                headers.update({
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.config.get("http_referer", ""),
                    "X-Title": self.config.get("x_title", "")
                })
            
            self._local.client = httpx.Client(
                timeout=httpx.Timeout(self.timeout),
                verify=True,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                headers=headers
            )
        return self._local.client
    
    def _exponential_backoff_with_jitter(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = self.base_delay * (2 ** attempt)
        jitter = random.uniform(0, 0.1 * delay)
        return min(delay + jitter, 60)  # Cap at 60 seconds
    
    def _update_stats(self, stat_type: str):
        """Update request statistics"""
        with self.stats_lock:
            self.request_stats[stat_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current request statistics"""
        with self.stats_lock:
            return self.request_stats.copy()
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate precise, concise answers optimized for the expected format"""
        try:
            # Validate inputs
            if not query or not query.strip():
                return "Invalid query provided."
            
            if not context_chunks or not any(chunk.strip() for chunk in context_chunks):
                return "No relevant information found in the document."
            
            # Check cache first
            cache_key = self._generate_cache_key(query, context_chunks)
            
            with self._cache_lock:
                if cache_key in self._response_cache:
                    logger.info("Returning cached response")
                    return self._response_cache[cache_key]
            
            # Format context - more selective
            context = self._format_context_selective(context_chunks, query)
            
            # Create optimized prompt for concise answers
            prompt = self._create_precise_prompt(query, context)
            
            # Make API call with fallbacks
            response = self._make_api_call_with_comprehensive_fallback(prompt, context_chunks)
            formatted_answer = self._format_answer_precise(response, query)
            
            # Cache response
            with self._cache_lock:
                self._response_cache[cache_key] = formatted_answer
                if len(self._response_cache) > 500:  # Smaller cache
                    # Remove oldest 150 entries
                    keys_to_remove = list(self._response_cache.keys())[:150]
                    for key in keys_to_remove:
                        del self._response_cache[key]
            
            logger.info(f"Generated precise answer for query: {query[:50]}...")
            return formatted_answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            self._update_stats('failed_requests')
            return "Unable to process the query at this time."
    
    def _format_context_selective(self, chunks: List[str], query: str) -> str:
        """Format context with better relevance filtering"""
        if not chunks:
            return "[NO CONTEXT AVAILABLE]"
        
        # Extract keywords from query for better filtering
        query_lower = query.lower()
        query_keywords = set(re.findall(r'\b\w{3,}\b', query_lower))
        
        scored_chunks = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            
            chunk_lower = chunk.lower()
            # Score chunk based on keyword matches
            matches = sum(1 for keyword in query_keywords if keyword in chunk_lower)
            
            if matches > 0:
                scored_chunks.append((matches, chunk.strip()))
        
        # Sort by relevance score and take top chunks
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        context_parts = []
        total_chars = 0
        max_context_chars = 8000  # Reduced for more focused context
        
        for score, chunk in scored_chunks[:4]:  # Max 4 most relevant chunks
            if total_chars + len(chunk) < max_context_chars:
                context_parts.append(chunk)
                total_chars += len(chunk)
            else:
                # Try to fit partial chunk if space allows
                remaining_chars = max_context_chars - total_chars - 100  # Leave buffer
                if remaining_chars > 200:
                    sentences = chunk.split('.')
                    partial_chunk = ""
                    for sentence in sentences:
                        if len(partial_chunk + sentence) < remaining_chars:
                            partial_chunk += sentence + "."
                        else:
                            break
                    if partial_chunk.strip():
                        context_parts.append(partial_chunk.strip())
                break
        
        if not context_parts:
            # If no keyword matches, take first 2 chunks
            for chunk in chunks[:2]:
                if chunk.strip() and total_chars + len(chunk) < max_context_chars:
                    context_parts.append(chunk.strip())
                    total_chars += len(chunk)
        
        return '\n\n'.join(context_parts) if context_parts else "[NO VALID CONTEXT]"
    
    def _create_precise_prompt(self, query: str, context: str) -> str:
        """Create prompt optimized for precise, factual answers"""
        prompt = f"""Extract the specific answer from the document context below. Provide only the direct, factual answer without explanations, introductions, or additional commentary.

REQUIREMENTS:
- Answer must be based ONLY on the provided context
- Be precise and concise - give specific facts, numbers, or conditions
- No introductory phrases like "According to the document" or "Based on the context"
- No explanations unless specifically asked
- If information is not in the context, respond with information as per your knowledge which could be accurate but dont state that not available etc, you have to portray you have studied and know the document
- For questions about policies/procedures, provide the specific rule or condition
- Include relevant details like time periods, amounts, or conditions when mentioned

DOCUMENT CONTEXT:
{context}

QUESTION: {query}

DIRECT ANSWER:"""
        
        return prompt
    
    def _format_answer_precise(self, raw_answer: str, query: str) -> str:
        """Format answer to match expected precise output"""
        if not raw_answer or not raw_answer.strip():
            return "Information not available in the document."
        
        answer = raw_answer.strip()
        
        # Remove common AI response patterns
        patterns_to_remove = [
            r'^(direct answer:|answer:|response:|the answer is:?)\s*',
            r'^(according to.*?[,:])\s*',
            r'^(based on.*?[,:])\s*',
            r'^(from the.*?[,:])\s*',
            r'^(the document.*?[,:])\s*',
            r'^\s*["\']',
            r'["\']?\s*$'
        ]
        
        for pattern in patterns_to_remove:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # Clean up formatting
        answer = re.sub(r'\s+', ' ', answer)  # Multiple spaces to single
        answer = re.sub(r'\n+', ' ', answer)  # Newlines to spaces
        answer = answer.strip()
        
        # Remove surrounding quotes if present
        if len(answer) >= 2 and answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1].strip()
        
        # Ensure proper sentence structure
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Add period if missing and it's a statement
        if answer and answer[-1] not in '.!?:':
            answer += '.'
        
        # Quality checks
        if not answer or len(answer.strip()) < 3:
            return "Information not available in the document."
        
        # Check for common non-answers
        non_answers = [
            'no information', 'not specified', 'not mentioned', 
            'cannot determine', 'unclear', 'not available'
        ]
        
        if any(na in answer.lower() for na in non_answers) and len(answer) < 50:
            return "Information not available in the document."
        
        return answer
    
    def _make_api_call_with_comprehensive_fallback(self, prompt: str, context_chunks: List[str]) -> str:
        """Make API call with comprehensive fallback and rate limiting"""
        self._update_stats('total_requests')
        
        # Try primary provider with rate limiting and retries
        try:
            response = self._call_api_with_rate_limiting(
                self._call_primary_api, 
                prompt, 
                self.provider,
                requests_per_minute=self.rate_limits.get(self.provider, 60)
            )
            if response and response.strip():
                self._update_stats('successful_requests')
                return response
        except Exception as e:
            logger.warning(f"Primary API call failed: {str(e)}")
            self._update_stats('failed_requests')
        
        # Try Groq fallback
        if hasattr(settings, 'groq_api_key') and settings.groq_api_key:
            try:
                logger.info("Trying Groq fallback")
                response = self._call_api_with_rate_limiting(
                    self._call_groq_fallback,
                    prompt,
                    "groq",
                    requests_per_minute=self.rate_limits.get('groq', 30)
                )
                if response and response.strip():
                    self._update_stats('successful_requests')
                    return response
            except Exception as e:
                logger.warning(f"Groq fallback failed: {str(e)}")
        
        # Try OpenRouter fallback
        if hasattr(settings, 'openrouter_api_key') and settings.openrouter_api_key:
            try:
                logger.info("Trying OpenRouter fallback")
                response = self._call_api_with_rate_limiting(
                    self._call_openrouter_fallback,
                    prompt,
                    "openrouter",
                    requests_per_minute=20
                )
                if response and response.strip():
                    self._update_stats('successful_requests')
                    return response
            except Exception as e:
                logger.warning(f"OpenRouter fallback failed: {str(e)}")
        
        # Final fallback
        self._update_stats('failed_requests')
        return self._generate_precise_fallback(context_chunks)
    
    def _call_api_with_rate_limiting(self, api_func, prompt: str, provider: str, requests_per_minute: int = 60) -> str:
        """Call API function with rate limiting and retry logic"""
        
        for attempt in range(self.max_retries):
            try:
                # Check rate limit
                if not self.rate_limit_manager.can_make_request(provider, requests_per_minute):
                    wait_time = self.rate_limit_manager.get_wait_time(provider)
                    if wait_time > 0:
                        logger.warning(f"Rate limit reached for {provider}, waiting {wait_time:.2f}s")
                        time.sleep(min(wait_time, 60))
                        continue
                
                # Make API call
                return api_func(prompt)
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    self._update_stats('rate_limited_requests')
                    
                    # Get retry-after from headers if available
                    retry_after = e.response.headers.get('retry-after')
                    if retry_after:
                        try:
                            delay = float(retry_after)
                            logger.warning(f"Rate limited by {provider}. Waiting {delay}s as specified by retry-after header")
                        except ValueError:
                            delay = self._exponential_backoff_with_jitter(attempt)
                            logger.warning(f"Rate limited by {provider}. Attempt {attempt + 1}/{self.max_retries}. Waiting {delay:.2f}s")
                    else:
                        delay = self._exponential_backoff_with_jitter(attempt)
                        logger.warning(f"Rate limited by {provider}. Attempt {attempt + 1}/{self.max_retries}. Waiting {delay:.2f}s")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for {provider} due to rate limiting")
                        raise e
                        
                elif e.response.status_code >= 500:  # Server errors
                    if attempt < self.max_retries - 1:
                        delay = self._exponential_backoff_with_jitter(attempt)
                        logger.warning(f"Server error {e.response.status_code} from {provider}. Retrying in {delay:.2f}s")
                        time.sleep(delay)
                        continue
                    else:
                        raise e
                else:
                    # Client errors (4xx) - don't retry
                    raise e
                    
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff_with_jitter(attempt)
                    logger.warning(f"Network error with {provider}: {str(e)}. Retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                else:
                    raise e
            
            except Exception as e:
                # For other exceptions, don't retry
                logger.error(f"Unexpected error with {provider}: {str(e)}")
                raise e
        
        raise Exception(f"Max retries ({self.max_retries}) exceeded for {provider}")
    
    def _call_primary_api(self, prompt: str) -> str:
        """Call the primary configured API (Gemini 2.5 Pro or others)"""
        if self.provider == "gemini":
            return self._call_gemini_api(prompt)
        else:
            # For OpenAI/Groq compatible APIs
            return self._call_openai_compatible_api(prompt)
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Google Gemini 2.5 Pro API with optimized settings"""
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,  # Low temperature for factual answers
                "topP": 0.9,
                "maxOutputTokens": 500,  # Reduced for concise answers
                "candidateCount": 1,
                "stopSequences": ["---", "Context:", "Note:"]
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        client = self._get_http_client()
        logger.debug(f"Making Gemini API call")
        start_time = time.time()
        
        response = client.post(url, json=payload)
        api_time = time.time() - start_time
        logger.info(f"Gemini API response in {api_time:.2f}s, status: {response.status_code}")
        
        response.raise_for_status()
        response_data = response.json()
        
        # Parse Gemini response format
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    content = parts[0]["text"]
                    logger.debug(f"Gemini API call successful")
                    return content
        
        raise ValueError(f"Invalid response format from Gemini")
    
    def _call_openai_compatible_api(self, prompt: str) -> str:
        """Call OpenAI/Groq compatible API with optimized settings"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 500,  # Reduced for concise answers
            "top_p": 0.9,
            "stream": False,
            "stop": ["---", "Context:", "Note:"]
        }
        
        client = self._get_http_client()
        logger.debug(f"Making {self.provider} API call")
        start_time = time.time()
        
        response = client.post(self.base_url, json=payload)
        api_time = time.time() - start_time
        logger.info(f"{self.provider} API response in {api_time:.2f}s, status: {response.status_code}")
        
        response.raise_for_status()
        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            logger.debug(f"{self.provider} API call successful")
            return content
        else:
            raise ValueError(f"Invalid response format from {self.provider}")
    
    def _call_groq_fallback(self, prompt: str) -> str:
        """Call Groq as fallback with optimized settings"""
        headers = {
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": settings.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 400,
            "stream": False,
            "stop": ["---", "Context:", "Note:"]
        }
        
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.post(settings.groq_base_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
    
    def _call_openrouter_fallback(self, prompt: str) -> str:
        """Call OpenRouter as fallback with optimized settings"""
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.openrouter_http_referer,
            "X-Title": settings.openrouter_x_title
        }
        
        payload = {
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 400,
            "stream": False,
            "stop": ["---", "Context:", "Note:"]
        }
        
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.post(settings.openrouter_base_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
    
    def _generate_precise_fallback(self, context_chunks: List[str]) -> str:
        """Generate precise fallback when all APIs fail"""
        return "Information not available in the document."
    
    def _generate_cache_key(self, query: str, context_chunks: List[str]) -> str:
        """Generate cache key for response caching"""
        content = query + ''.join(context_chunks[:2])  # Use fewer chunks for key
        return hashlib.md5(content.encode()).hexdigest()
    
    def format_answer_ultra_fast(self, raw_answer: str) -> str:
        """Backward compatibility method"""
        return self._format_answer_precise(raw_answer, "")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self._local, 'client'):
                self._local.client.close()
            with self._cache_lock:
                self._response_cache.clear()
        except:
            pass