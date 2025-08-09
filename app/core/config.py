from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    api_title: str = "Intelligent Document Query System"
    api_version: str = "2.0.0"
    api_description: str = "AI-powered document processing and query system with unlimited responses"
    
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    
    max_file_size: int = 500 * 1024 * 1024  # Increased to 500MB
    allowed_extensions: list = [".pdf", ".docx", ".txt"]
    chunk_size: int = 2000  # Larger chunks for more context
    chunk_overlap: int = 200  # More overlap
    max_context_chunks: int = 20  # More context chunks
    
    # Primary LLM Configuration - OpenRouter Mistral (UNLIMITED)
    llm_provider: str = "openrouter"
    openrouter_api_key: str
    openrouter_model: str = "mistralai/codestral-2508"
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_http_referer: str = "https://your-domain.com"
    openrouter_x_title: str = "Document Query System"
    
    # Rate Limiting Configuration
    openrouter_requests_per_minute: int = 20
    groq_requests_per_minute: int = 30
    gemini_requests_per_minute: int = 60
    
    # Retry and Error Handling Configuration
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 120.0  # Increased for unlimited responses
    enable_exponential_backoff: bool = True
    respect_retry_after_header: bool = True
    
    # First Fallback - Groq (UNLIMITED)
    groq_api_key: Optional[str] = None
    groq_model: Optional[str] = "llama-3.3-70b-versatile"
    groq_base_url: Optional[str] = "https://api.groq.com/openai/v1/chat/completions"
    groq_retry_attempts: Optional[int] = 2
    groq_retry_delay: Optional[float] = 1.0
    
    # Second Fallback - Google Gemini (UNLIMITED)
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = "gemini-2.0-flash-exp"
    gemini_base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/models"

    # Legacy OpenAI configuration
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = "gpt-4o"
    openai_base_url: Optional[str] = "https://api.openai.com/v1/chat/completions"
    openai_requests_per_minute: int = 60
    
    # UNLIMITED LLM Parameters
    confidence_threshold: float = 0.8
    max_tokens: Optional[int] = None  # NO TOKEN LIMIT
    temperature: float = 0.1
    top_p: float = 0.9
    request_timeout: int = 300  # 5 minutes for unlimited responses
    
    # Processing Parameters - Enhanced for unlimited processing
    max_pages_per_document: int = 1000  # Increased limit
    processing_timeout_per_page: float = 1.0  # More time per page
    parallel_processing: bool = True
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dimension: int = 768
    embedding_batch_size: int = 128  # Larger batches
    
    # Vector Database Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_env: Optional[str] = None
    pinecone_index_name: str = "document-embeddings-v2"
    
    # Enhanced Local Vector Storage
    vector_storage_path: str = "data/vectors"
    cache_size: int = 10000  # Larger cache
    enable_compression: bool = True
    
    # UNLIMITED Search Configuration
    similarity_threshold: float = 0.6  # Lower threshold for more results
    max_search_results: int = 50  # Much more results
    rerank_enabled: bool = True
    hybrid_search_enabled: bool = True
    semantic_search_weight: float = 0.7
    keyword_search_weight: float = 0.3
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"
    enable_detailed_logging: bool = True
    
    # CORS Configuration
    cors_origins: list = ["*"]
    cors_methods: list = ["GET", "POST", "PUT", "DELETE"]
    cors_headers: list = ["*"]
    
    # UNLIMITED Performance Configuration
    api_request_timeout: int = 600  # 10 minutes for unlimited responses
    batch_size: int = 500  # Larger batches
    max_concurrent_requests: int = 50  # More concurrent requests
    enable_response_caching: bool = True
    cache_ttl: int = 7200  # 2 hour cache for unlimited responses
    
    # Rate Limiting Monitoring
    enable_rate_limit_monitoring: bool = True
    log_rate_limit_events: bool = True
    rate_limit_alert_threshold: int = 10
    
    # Circuit Breaker Configuration (Disabled for unlimited mode)
    enable_circuit_breaker: bool = False
    circuit_breaker_failure_threshold: int = 10
    circuit_breaker_timeout: int = 120
    circuit_breaker_recovery_timeout: int = 60
    
    # Webhook Configuration
    webhook_secret: Optional[str] = None
    webhook_timeout: int = 60  # Increased timeout
    
    # Development Settings
    debug: bool = False
    reload: bool = False
    enable_metrics: bool = True
    
    # UNLIMITED MODE FLAGS
    unlimited_mode: bool = True
    remove_token_limits: bool = True
    remove_time_limits: bool = True
    comprehensive_answers: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_llm_config(self) -> dict:
        """Get LLM configuration dictionary for unlimited responses"""
        base_config = {
            "max_tokens": None,  # NO TOKEN LIMIT
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.request_timeout,
            "unlimited_mode": True
        }
        
        if self.llm_provider == "openrouter":
            return {
                "provider": "openrouter",
                "api_key": self.openrouter_api_key,
                "model": self.openrouter_model,
                "base_url": self.openrouter_base_url,
                "http_referer": self.openrouter_http_referer,
                "x_title": self.openrouter_x_title,
                **base_config
            }
        elif self.llm_provider == "groq":
            return {
                "provider": "groq",
                "api_key": self.groq_api_key,
                "model": self.groq_model,
                "base_url": self.groq_base_url,
                "retry_attempts": self.groq_retry_attempts,
                "retry_delay": self.groq_retry_delay,
                **base_config
            }
        elif self.llm_provider == "gemini":
            return {
                "provider": "gemini",
                "api_key": self.gemini_api_key,
                "model": self.gemini_model,
                "base_url": self.gemini_base_url,
                **base_config
            }
        else:  # fallback to openai
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "base_url": self.openai_base_url,
                **base_config
            }
    
    def get_rate_limit_config(self) -> dict:
        """Get rate limiting configuration for unlimited mode"""
        return {
            "openrouter_rpm": self.openrouter_requests_per_minute,
            "groq_rpm": self.groq_requests_per_minute,
            "gemini_rpm": self.gemini_requests_per_minute,
            "openai_rpm": self.openai_requests_per_minute,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "exponential_backoff": self.enable_exponential_backoff,
            "respect_retry_after": self.respect_retry_after_header,
            "monitoring_enabled": self.enable_rate_limit_monitoring,
            "unlimited_mode": self.unlimited_mode
        }
    
    def get_embedding_config(self) -> dict:
        """Get enhanced embedding configuration for unlimited mode"""
        return {
            "model": self.embedding_model,
            "dimension": self.embedding_dimension,
            "batch_size": self.embedding_batch_size,
            "unlimited_mode": self.unlimited_mode
        }
    
    def get_vector_config(self) -> dict:
        """Get enhanced vector database configuration for unlimited mode"""
        return {
            "pinecone_api_key": self.pinecone_api_key,
            "pinecone_env": self.pinecone_env,
            "index_name": self.pinecone_index_name,
            "storage_path": self.vector_storage_path,
            "similarity_threshold": self.similarity_threshold,
            "enable_compression": self.enable_compression,
            "unlimited_results": True,
            "max_results": self.max_search_results
        }
     
# Initialize settings instance
settings = Settings()

# Create necessary directories
os.makedirs("data/vectors", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("temp", exist_ok=True)
os.makedirs("cache", exist_ok=True)

# Validate critical configuration on startup
def validate_config():
    """Validate critical configuration settings for unlimited mode"""
    if settings.llm_provider == "openrouter" and not settings.openrouter_api_key:
        raise ValueError("OpenRouter API key is required. Please set OPENROUTER_API_KEY in your .env file")
    
    if settings.llm_provider == "groq" and not settings.groq_api_key:
        raise ValueError("Groq API key is required when using Groq provider")
    
    if settings.llm_provider == "gemini" and not settings.gemini_api_key:
        raise ValueError("Gemini API key is required when using Gemini provider")
        
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        raise ValueError("OpenAI API key is required when using OpenAI provider")
    
    # Validate unlimited mode settings
    if settings.unlimited_mode:
        if settings.max_tokens is not None:
            print("Warning: max_tokens is set but unlimited_mode is enabled. Removing token limits.")
            settings.max_tokens = None
        
        if settings.request_timeout < 300:
            print("Warning: request_timeout is low for unlimited mode. Consider increasing to 300+ seconds.")
    
    print(f"Configuration validated successfully - UNLIMITED MODE ENABLED")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Model: {settings.openrouter_model if settings.llm_provider == 'openrouter' else settings.groq_model if settings.llm_provider == 'groq' else settings.gemini_model if settings.llm_provider == 'gemini' else settings.openai_model}")
    print(f"Token Limits: DISABLED (Unlimited responses)")
    print(f"Time Limits: Extended ({settings.request_timeout}s timeout)")
    print(f"Max Context Chunks: {settings.max_context_chunks}")
    print(f"Max Search Results: {settings.max_search_results}")
    print(f"Rate Limits: OpenRouter({settings.openrouter_requests_per_minute}/min), Groq({settings.groq_requests_per_minute}/min), Gemini({settings.gemini_requests_per_minute}/min)")
    print(f"Debug Mode: {settings.debug}")
    print(f"Comprehensive Answers: {settings.comprehensive_answers}")

# Run validation
validate_config()