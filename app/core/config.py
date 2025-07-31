import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # api config
    api_title: str = "intelligent doc query system"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # llm config
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-pro"
    
    # vector db config
    pinecone_api_key: str
    pinecone_env: str
    pinecone_index: str = "doc-index"
    
    # embedding config
    embedding_model: str = "all-mpnet-base-v2"
    rerank_model: str = "ms-marco-MiniLM-L-6-v2"
    
    # processing config
    chunk_size: int = 400
    chunk_overlap: int = 50
    max_context_chunks: int = 5
    confidence_threshold: float = 0.7
    
    # server config
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
