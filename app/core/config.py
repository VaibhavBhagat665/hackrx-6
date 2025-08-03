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
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_chunks: int = 5
    
    # LLM Configuration
    gemini_api_key: str
    gemini_model: str = "gemini-pro"
    confidence_threshold: float = 0.7
    max_tokens: int = 1000
    temperature: float = 0.3
    
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    pinecone_api_key: Optional[str] = None
    pinecone_env: Optional[str] = None
    pinecone_index_name: str = "document-embeddings"
    
    vector_storage_path: str = "data/vectors"
    cache_size: int = 1000
    
    similarity_threshold: float = 0.7
    max_search_results: int = 10
    rerank_enabled: bool = True
    
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_rotation: str = "1 day"
    log_retention: str = "30 days"
    
    cors_origins: list = ["*"]
    cors_methods: list = ["GET", "POST", "PUT", "DELETE"]
    cors_headers: list = ["*"]
    
    request_timeout: int = 300 
    batch_size: int = 100
    max_concurrent_requests: int = 10
    
    webhook_secret: Optional[str] = None
    webhook_timeout: int = 30
    
    debug: bool = False
    reload: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
     
settings = Settings()

os.makedirs("data/vectors", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("temp", exist_ok=True)
