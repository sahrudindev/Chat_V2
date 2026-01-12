"""
Application configuration using Pydantic Settings.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "financial_data"
    
    # AI Service (Local, accessed via host.docker.internal from Docker)
    ai_service_url: str = "http://localhost:8001"
    
    # RAG Configuration
    rag_top_k: int = 5
    rag_score_threshold: float = 0.3
    
    # Application
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
