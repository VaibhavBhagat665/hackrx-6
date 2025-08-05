from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    api_title: str = "Intelligent Document Query System"
    api_version: str = "1.0.0"
    api_description: str = "AI-powered document processing and query system"
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    max_file_size: int = 100 * 1024 * 1024 
    allowed_extensions: list = [".pdf", ".docx", ".txt"]
    chunk_size: int = 300  # Reduced from 1000 for faster processing
    chunk_overlap: int = 30  # Reduced from 200 for faster processing
    max_context_chunks: int = 3  # Reduced from 5 for faster processing
    
    # LLM Configuration - Updated for OpenRouter + Mistral
    llm_provider: str = "openrouter"
    openrouter_api_key: str
    openrouter_model: str = "mistralai/codestral-2508"
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    
    # Legacy Gemini configuration (kept for backward compatibility, can be removed)
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = "gemini-pro"
    
    # LLM Parameters
    confidence_threshold: float = 0.7
    max_tokens: int = 1000
    temperature: float = 0.3
    top_p: float = 0.9
    request_timeout: int = 60  # Timeout for LLM API calls
    
    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Vector Database Configuration
    pinecone_api_key: Optional[str] = None
    pinecone_env: Optional[str] = None
    pinecone_index_name: str = "document-embeddings"
    
    # Local Vector Storage
    vector_storage_path: str = "data/vectors"
    cache_size: int = 1000
    
    # Search Configuration
    similarity_threshold: float = 0.7
    max_search_results: int = 10
    rerank_enabled: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"
    
    # CORS Configuration
    cors_origins: list = ["*"]
    cors_methods: list = ["GET", "POST", "PUT", "DELETE"]
    cors_headers: list = ["*"]
    
    # Performance Configuration
    api_request_timeout: int = 300  # General API timeout
    batch_size: int = 100
    max_concurrent_requests: int = 10
    
    # Webhook Configuration
    webhook_secret: Optional[str] = None
    webhook_timeout: int = 30
    
    # OpenRouter Specific Configuration
    openrouter_http_referer: str = "https://your-domain.com"  # Optional: for analytics
    openrouter_x_title: str = "Document Query System"  # Optional: for analytics
    
    # Development Settings
    debug: bool = False
    reload: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_llm_config(self) -> dict:
        """Get LLM configuration dictionary for easy access"""
        return {
            "provider": self.llm_provider,
            "api_key": self.openrouter_api_key,
            "model": self.openrouter_model,
            "base_url": self.openrouter_base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.request_timeout,
            "http_referer": self.openrouter_http_referer,
            "x_title": self.openrouter_x_title
        }
    
    def get_embedding_config(self) -> dict:
        """Get embedding configuration dictionary"""
        return {
            "model": self.embedding_model,
            "dimension": self.embedding_dimension
        }
    
    def get_vector_config(self) -> dict:
        """Get vector database configuration dictionary"""
        return {
            "pinecone_api_key": self.pinecone_api_key,
            "pinecone_env": self.pinecone_env,
            "index_name": self.pinecone_index_name,
            "storage_path": self.vector_storage_path,
            "similarity_threshold": self.similarity_threshold
        }
     
# Initialize settings instance
settings = Settings()

# Create necessary directories
os.makedirs("data/vectors", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Validate critical configuration on startup
def validate_config():
    """Validate critical configuration settings"""
    if not settings.openrouter_api_key:
        raise ValueError("OpenRouter API key is required. Please set OPENROUTER_API_KEY in your .env file")
    
    if settings.llm_provider == "openrouter" and not settings.openrouter_model:
        raise ValueError("OpenRouter model must be specified when using OpenRouter provider")
    
    print(f"✅ Configuration validated successfully")
    print(f"📋 LLM Provider: {settings.llm_provider}")
    print(f"🤖 Model: {settings.openrouter_model}")
    print(f"🔧 Debug Mode: {settings.debug}")

# Run validation
validate_config()