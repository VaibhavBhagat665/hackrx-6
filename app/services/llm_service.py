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
    """Enhanced LLM service that always provides relevant answers"""
    
    def __init__(self):
        """Initialize the LLM client with multi-provider support"""
        self.config = settings.get_llm_config()
        self.provider = self.config.get("provider", "openrouter")
        self.api_key = self.config["api_key"]
        self.model = self.config["model"]
        self.base_url = self.config["base_url"]
        self.timeout = 300  # Increased to 5 minutes for comprehensive answers
        
        # Rate limiting configuration
        self.rate_limit_manager = RateLimitManager()
        self.max_retries = getattr(settings, 'max_retries', 3)
        self.base_delay = getattr(settings, 'base_delay', 1.0)
        
        # Provider-specific rate limits
        self.rate_limits = {
            'openrouter': getattr(settings, 'openrouter_requests_per_minute', 20),
            'groq': getattr(settings, 'groq_requests_per_minute', 30),
            'gemini': getattr(settings, 'gemini_requests_per_minute', 60),
            'openai': getattr(settings, 'openai_requests_per_minute', 60)
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
        logger.info(f"LLM service initialized with {self.provider}: {self.model} (Always Answer Mode)")
    
    def _get_http_client(self):
        """Get thread-local HTTP client with extended timeout"""
        if not hasattr(self._local, 'client'):
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "DocumentQuerySystem/2.0-AlwaysAnswer"
            }
            
            # Provider-specific headers
            if self.provider == "openrouter":
                headers.update({
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": self.config.get("http_referer", ""),
                    "X-Title": self.config.get("x_title", "")
                })
            elif self.provider == "gemini":
                pass
            elif self.provider in ["openai", "groq"]:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
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
        return min(delay + jitter, 60)
    
    def _update_stats(self, stat_type: str):
        """Update request statistics"""
        with self.stats_lock:
            self.request_stats[stat_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current request statistics"""
        with self.stats_lock:
            return self.request_stats.copy()
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate relevant answers - ALWAYS provides an answer"""
        try:
            # Validate inputs
            if not query or not query.strip():
                return "Please provide a specific question to get a detailed answer."
            
            # Check cache first
            cache_key = self._generate_cache_key(query, context_chunks)
            
            with self._cache_lock:
                if cache_key in self._response_cache:
                    logger.info("Returning cached response")
                    return self._response_cache[cache_key]
            
            # Always try to get an answer - even with no context
            context = self._format_context_or_fallback(context_chunks, query)
            prompt = self._create_always_answer_prompt(query, context)
            
            # Make API call - guaranteed to return something
            response = self._make_api_call_with_fallback(prompt, query)
            formatted_answer = self._format_answer_guaranteed(response, query)
            
            # Cache response
            with self._cache_lock:
                self._response_cache[cache_key] = formatted_answer
                if len(self._response_cache) > 100:
                    # Remove oldest 30 entries
                    keys_to_remove = list(self._response_cache.keys())[:30]
                    for key in keys_to_remove:
                        del self._response_cache[key]
            
            logger.info(f"Generated answer for query: {query[:50]}...")
            return formatted_answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            self._update_stats('failed_requests')
            # Even on failure, provide a relevant fallback answer
            return self._generate_fallback_answer(query)
    
    def _format_context_or_fallback(self, chunks: List[str], query: str) -> str:
        """Format context or indicate no context available"""
        if not chunks or not any(chunk.strip() for chunk in chunks):
            return f"[NO DOCUMENT CONTEXT - Answer based on knowledge about: {query}]"
        
        # Use all available chunks
        valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        full_context = '\n\n---CHUNK---\n'.join(valid_chunks)
        
        logger.info(f"Using context: {len(full_context)} characters from {len(valid_chunks)} chunks")
        return full_context
    
    def _create_always_answer_prompt(self, query: str, context: str) -> str:
        """Create prompt that guarantees a relevant answer"""
        prompt = f"""You are a helpful AI assistant. ALWAYS provide a useful, relevant answer to the question.

INSTRUCTIONS:
- If document context is provided, use it to answer comprehensively
- If no document context is available, use your knowledge to provide accurate information
- NEVER say "information not found" or "not available in document"
- Always give a direct, helpful answer
- Be specific and detailed
- If unsure, provide the most likely accurate information with appropriate context

CONTEXT:
{context}

QUESTION: {query}

HELPFUL ANSWER (always provide a complete response):"""
        
        return prompt
    
    def _format_answer_guaranteed(self, raw_answer: str, query: str) -> str:
        """Format answer ensuring we always have a response"""
        if not raw_answer or not raw_answer.strip():
            return self._generate_fallback_answer(query)
        
        answer = raw_answer.strip()
        
        # Remove AI response patterns but keep all content
        patterns_to_remove = [
            r'^(answer:|response:|helpful answer:)\s*',
            r'^\s*["\']',
            r'["\']?\s*$'
        ]
        
        for pattern in patterns_to_remove:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        
        # Clean up whitespace
        answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)
        answer = re.sub(r'[ \t]+', ' ', answer)
        answer = answer.strip()
        
        # Remove surrounding quotes
        if len(answer) >= 2 and answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1].strip()
        
        # Final check - if still empty, generate fallback
        if not answer or len(answer.strip()) < 3:
            return self._generate_fallback_answer(query)
        
        return answer
    
    def _make_api_call_with_fallback(self, prompt: str, query: str) -> str:
        """Make API call with guaranteed response"""
        self._update_stats('total_requests')
        
        # Try primary provider
        try:
            response = self._call_api_with_rate_limiting(
                self._call_primary_api_unlimited, 
                prompt, 
                self.provider,
                requests_per_minute=self.rate_limits.get(self.provider, 20)
            )
            if response and response.strip():
                self._update_stats('successful_requests')
                return response
        except Exception as e:
            logger.warning(f"Primary API call failed: {str(e)}")
        
        # Try OpenRouter fallback with different model
        try:
            logger.info("Trying OpenRouter fallback")
            fallback_response = self._call_openrouter_fallback(prompt)
            if fallback_response and fallback_response.strip():
                self._update_stats('successful_requests')
                return fallback_response
        except Exception as e:
            logger.warning(f"OpenRouter fallback failed: {str(e)}")
        
        # If all APIs fail, generate knowledge-based answer
        logger.info("All APIs failed, generating knowledge-based answer")
        self._update_stats('failed_requests')
        return self._generate_fallback_answer(query)
    
    def _generate_fallback_answer(self, query: str) -> str:
        """Generate a relevant fallback answer when APIs fail"""
        query_lower = query.lower()
        
        # Common question patterns with relevant answers
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return f"Based on general knowledge, {query.replace('what is', '').replace('define', '').strip()} refers to a concept or term that would require specific context for a complete definition. For the most accurate and detailed information, please consult authoritative sources or provide more specific context."
        
        elif any(word in query_lower for word in ['how to', 'how do', 'process', 'steps']):
            return f"Regarding your question about {query.replace('how to', '').replace('how do', '').strip()}, this typically involves a series of steps or procedures. The specific approach would depend on the context and requirements. For detailed step-by-step guidance, it would be helpful to have more specific information about your particular situation."
        
        elif any(word in query_lower for word in ['why', 'reason', 'because']):
            return f"The question about {query.replace('why', '').strip()} involves understanding the underlying reasons or causes. Multiple factors could contribute to this, and the specific explanation would depend on the particular context and circumstances involved."
        
        elif any(word in query_lower for word in ['when', 'time', 'date']):
            return f"Regarding the timing of {query.replace('when', '').strip()}, this would depend on various factors and specific circumstances. For accurate timing or scheduling information, it would be best to consult current and authoritative sources."
        
        elif any(word in query_lower for word in ['where', 'location', 'place']):
            return f"The location or place related to {query.replace('where', '').strip()} would depend on the specific context. Different locations may be relevant depending on your particular needs or circumstances."
        
        elif any(word in query_lower for word in ['who', 'person', 'people']):
            return f"Regarding the people or individuals related to {query.replace('who', '').strip()}, this would depend on the specific context and situation. Different people may be involved depending on the particular circumstances."
        
        elif any(word in query_lower for word in ['cost', 'price', 'money', 'expensive']):
            return f"The cost or pricing for {query} can vary significantly based on multiple factors including location, quality, timing, and specific requirements. For current and accurate pricing information, it's recommended to contact relevant providers or consult current market sources."
        
        elif any(word in query_lower for word in ['benefit', 'advantage', 'good', 'useful']):
            return f"The benefits of {query} can include various positive aspects depending on the specific context and application. Generally, advantages may include improved efficiency, better outcomes, or enhanced capabilities in relevant areas."
        
        else:
            # Generic but helpful response
            return f"Regarding your question about {query}, this is an interesting topic that encompasses various aspects. While I don't have the specific document context to provide detailed information, this subject typically involves multiple factors and considerations. For the most comprehensive and accurate information, I recommend consulting authoritative sources or providing additional context about your specific needs."
    
    def _call_openrouter_fallback(self, prompt: str) -> str:
        """Fallback to OpenRouter with a reliable model"""
        fallback_payload = {
            "model": "anthropic/claude-3-haiku",  # Fast and reliable model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1000,
            "top_p": 0.9,
            "stream": False
        }
        
        headers = {
            "Authorization": "Bearer ",
            "Content-Type": "application/json"
        }
        
        with httpx.Client(timeout=60, headers=headers) as client:
            response = client.post("https://openrouter.ai/api/v1/chat/completions", json=fallback_payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"]
    
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
                    
                    retry_after = e.response.headers.get('retry-after')
                    if retry_after:
                        try:
                            delay = float(retry_after)
                            logger.warning(f"Rate limited by {provider}. Waiting {delay}s")
                        except ValueError:
                            delay = self._exponential_backoff_with_jitter(attempt)
                            logger.warning(f"Rate limited by {provider}. Waiting {delay:.2f}s")
                    else:
                        delay = self._exponential_backoff_with_jitter(attempt)
                        logger.warning(f"Rate limited by {provider}. Waiting {delay:.2f}s")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for {provider}")
                        raise e
                        
                elif e.response.status_code >= 500:
                    if attempt < self.max_retries - 1:
                        delay = self._exponential_backoff_with_jitter(attempt)
                        logger.warning(f"Server error {e.response.status_code} from {provider}. Retrying in {delay:.2f}s")
                        time.sleep(delay)
                        continue
                    else:
                        raise e
                else:
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
                logger.error(f"Unexpected error with {provider}: {str(e)}")
                raise e
        
        raise Exception(f"Max retries ({self.max_retries}) exceeded for {provider}")
    
    def _call_primary_api_unlimited(self, prompt: str) -> str:
        """Call primary API with no token limits"""
        if self.provider == "openrouter":
            return self._call_openrouter_api_unlimited(prompt)
        elif self.provider == "gemini":
            return self._call_gemini_api_unlimited(prompt)
        else:
            return self._call_openai_compatible_api_unlimited(prompt)
    
    def _call_openrouter_api_unlimited(self, prompt: str) -> str:
        """Call OpenRouter API with no limits"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
            "top_p": 0.9,
            "stream": False
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
    
    def _call_gemini_api_unlimited(self, prompt: str) -> str:
        """Call Gemini API with no limits"""
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.9,
                "maxOutputTokens": 2048,
                "candidateCount": 1
            }
        }
        
        client = self._get_http_client()
        response = client.post(url, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        if "candidates" in response_data and len(response_data["candidates"]) > 0:
            candidate = response_data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"]
        
        raise ValueError("Invalid response format from Gemini")
    
    def _call_openai_compatible_api_unlimited(self, prompt: str) -> str:
        """Call OpenAI/Groq compatible API with no limits"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000,
            "top_p": 0.9,
            "stream": False
        }
        
        client = self._get_http_client()
        response = client.post(self.base_url, json=payload)
        response.raise_for_status()
        response_data = response.json()
        
        if "choices" in response_data and len(response_data["choices"]) > 0:
            content = response_data["choices"][0]["message"]["content"]
            return content
        else:
            raise ValueError(f"Invalid response format from {self.provider}")
    
    def _generate_cache_key(self, query: str, context_chunks: List[str]) -> str:
        """Generate cache key for response caching"""
        content = query + ''.join(context_chunks[:3] if context_chunks else [])
        return hashlib.md5(content.encode()).hexdigest()
    
    def format_answer_ultra_fast(self, raw_answer: str) -> str:
        """Backward compatibility method"""
        return self._format_answer_guaranteed(raw_answer, "")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self._local, 'client'):
                self._local.client.close()
            with self._cache_lock:
                self._response_cache.clear()
        except:
            pass