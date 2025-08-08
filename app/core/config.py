from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    api_title: str = "Intelligent Document Query System"
    api_version: str = "2.0.0"
    api_description: str = "AI-powered document processing and query system with Gemini 2.5 Pro integration"
    
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    
    max_file_size: int = 200 * 1024 * 1024  # Increased to 200MB for larger docs
    allowed_extensions: list = [".pdf", ".docx", ".txt"]
    chunk_size: int = 800  # Increased for better context
    chunk_overlap: int = 100  # Increased overlap for better continuity
    max_context_chunks: int = 6  # More context for better answers
    
    # Primary LLM Configuration - Google Gemini 2.5 Pro
    llm_provider: str = "gemini"
    gemini_api_key: str  # Will use your GEMINI_API_KEY from env
    gemini_model: str = "gemini-2.0-flash-exp"  # Latest Gemini 2.0 Flash
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/models"
    
    # Rate Limiting Configuration
    gemini_requests_per_minute: int = 60
    groq_requests_per_minute: int = 30
    openrouter_requests_per_minute: int = 20
    
    # Retry and Error Handling Configuration
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    enable_exponential_backoff: bool = True
    respect_retry_after_header: bool = True
    
    # Fallback configurations
    groq_api_key: Optional[str] = None
    groq_model: Optional[str] = "llama-3.3-70b-versatile"
    groq_base_url: Optional[str] = "https://api.groq.com/openai/v1/chat/completions"
    groq_retry_attempts: Optional[int] = 2
    groq_retry_delay: Optional[float] = 1.0
    
    openrouter_api_key: Optional[str] = None
    openrouter_model: Optional[str] = "mistralai/codestral-2508"
    openrouter_base_url: Optional[str] = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_http_referer: str = "https://your-domain.com"
    openrouter_x_title: str = "Document Query System"

    # Legacy OpenAI configuration (kept for backward compatibility)
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = "gpt-4o"
    openai_base_url: Optional[str] = "https://api.openai.com/v1/chat/completions"
    openai_requests_per_minute: int = 60
    
    # Enhanced LLM Parameters for Gemini 2.5 Pro
    confidence_threshold: float = 0.8
    max_tokens: int = 1000  # Increased for more comprehensive answers
    temperature: float = 0.1  # Lower for more consistent, factual responses
    top_p: float = 0.9
    request_timeout: int = 45  # Increased timeout for better responses
    
    # Processing Parameters - Enhanced for large documents
    max_pages_per_document: int = 200
    processing_timeout_per_page: float = 0.2
    parallel_processing: bool = True
    
    # Embedding Configuration - Enhanced for better accuracy
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"  # Better model
    embedding_dimension: int = 768  # Higher dimension for better accuracy
    embedding_batch_size: int = 64
    
    # Vector Database Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_env: Optional[str] = None
    pinecone_index_name: str = "document-embeddings-v2"
    
    # Enhanced Local Vector Storage
    vector_storage_path: str = "data/vectors"
    cache_size: int = 5000  # Increased cache
    enable_compression: bool = True
    
    # Enhanced Search Configuration
    similarity_threshold: float = 0.65  # Slightly lower for more results
    max_search_results: int = 12  # More results for better context
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
    
    # Enhanced Performance Configuration
    api_request_timeout: int = 180  # Increased for complex queries
    batch_size: int = 200
    max_concurrent_requests: int = 20
    enable_response_caching: bool = True
    cache_ttl: int = 3600  # 1 hour cache
    
    # Rate Limiting Monitoring
    enable_rate_limit_monitoring: bool = True
    log_rate_limit_events: bool = True
    rate_limit_alert_threshold: int = 10  # Alert after 10 consecutive rate limits
    
    # Circuit Breaker Configuration (Advanced)
    enable_circuit_breaker: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_timeout: int = 60  # seconds
    circuit_breaker_recovery_timeout: int = 30  # seconds
    
    # Webhook Configuration
    webhook_secret: Optional[str] = None
    webhook_timeout: int = 30
    
    # Development Settings
    debug: bool = False
    reload: bool = False
    enable_metrics: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_llm_config(self) -> dict:
        """Get LLM configuration dictionary for Gemini 2.5 Pro"""
        base_config = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.request_timeout
        }
        
        if self.llm_provider == "gemini":
            return {
                "provider": "gemini",
                "api_key": self.gemini_api_key,
                "model": self.gemini_model,
                "base_url": self.gemini_base_url,
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
        elif self.llm_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "base_url": self.openai_base_url,
                **base_config
            }
        else:  # fallback to openrouter
            return {
                "provider": "openrouter",
                "api_key": self.openrouter_api_key,
                "model": self.openrouter_model,
                "base_url": self.openrouter_base_url,
                "http_referer": self.openrouter_http_referer,
                "x_title": self.openrouter_x_title,
                **base_config
            }
    
    def get_rate_limit_config(self) -> dict:
        """Get rate limiting configuration"""
        return {
            "gemini_rpm": self.gemini_requests_per_minute,
            "groq_rpm": self.groq_requests_per_minute,
            "openrouter_rpm": self.openrouter_requests_per_minute,
            "openai_rpm": self.openai_requests_per_minute,
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "exponential_backoff": self.enable_exponential_backoff,
            "respect_retry_after": self.respect_retry_after_header,
            "monitoring_enabled": self.enable_rate_limit_monitoring
        }
    
    def get_embedding_config(self) -> dict:
        """Get enhanced embedding configuration"""
        return {
            "model": self.embedding_model,
            "dimension": self.embedding_dimension,
            "batch_size": self.embedding_batch_size
        }
    
    def get_vector_config(self) -> dict:
        """Get enhanced vector database configuration"""
        return {
            "pinecone_api_key": self.pinecone_api_key,
            "pinecone_env": self.pinecone_env,
            "index_name": self.pinecone_index_name,
            "storage_path": self.vector_storage_path,
            "similarity_threshold": self.similarity_threshold,
            "enable_compression": self.enable_compression
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
    """Validate critical configuration settings"""
    if settings.llm_provider == "gemini" and not settings.gemini_api_key:
        raise ValueError("Gemini API key is required. Please set GEMINI_API_KEY in your .env file")
    
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        raise ValueError("OpenAI API key is required when using OpenAI provider")
    
    if settings.llm_provider == "groq" and not settings.groq_api_key:
        raise ValueError("Groq API key is required when using Groq provider")
        
    if settings.llm_provider == "openrouter" and not settings.openrouter_api_key:
        raise ValueError("OpenRouter API key is required when using OpenRouter provider")
    
    # Validate rate limiting settings
    if settings.gemini_requests_per_minute > 10000:
        logger.warning("Gemini requests per minute seems very high. Consider checking your tier limits.")
    
    if settings.max_retries > 10:
        logger.warning("Max retries is very high. This may cause long delays.")
    
    print(f"Configuration validated successfully")
    print(f"LLM Provider: {settings.llm_provider}")
    print(f"Model: {settings.gemini_model if settings.llm_provider == 'gemini' else settings.groq_model if settings.llm_provider == 'groq' else settings.openai_model if settings.llm_provider == 'openai' else settings.openrouter_model}")
    print(f"Rate Limits: Gemini({settings.gemini_requests_per_minute}/min), Groq({settings.groq_requests_per_minute}/min), OpenRouter({settings.openrouter_requests_per_minute}/min)")
    print(f"Debug Mode: {settings.debug}")
    print(f"Enhanced Features: Enabled")

# Run validation
validate_config()