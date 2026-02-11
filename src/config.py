"""
config.py

Centralized configuration management for the GenAI RAG & Agent application.

This module uses Pydantic BaseSettings to:
- Load configuration from environment variables and .env files.
- Provide type validation and defaults for all application settings.
- Implement a singleton pattern via lru_cache for memory efficiency.

Key considerations for developers:
- All sensitive credentials (API keys) should be stored in a .env file, not hardcoded.
- The Settings class follows SOLID principles: Single Responsibility (config only),
  Open/Closed (extendable via inheritance), and Dependency Inversion (injected via settings).
- Environment variables override default values in the Settings class.
- The get_settings() function is cached; it returns the same Settings instance on subsequent calls.
- If settings need to be reloaded (e.g., in tests), clear the cache with get_settings.cache_clear().
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):

    """
    Encapsulates all configuration and provides a single source of truth.
    """
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # LLM Settings
    gemini_api_key: str = ""
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.4

    
    # RAG & Embeddings
    top_k: int = 4
    data_path: str = "data"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    huggingface_api_token: str = ""
    
    # Vector Store Settings
    pinecone_api_key: str = ""
    pinecone_index_name: str = "genai-rag-agent"
    vector_dimension: int = 384  # Default for sentence-transformers/all-MiniLM-L6-v2
    

@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings class (Singleton pattern).
    """
    s = Settings()
    # Explicitly set environment variables for third-party libs like LangChain
    if s.gemini_api_key:
        os.environ["GOOGLE_API_KEY"] = s.gemini_api_key
    if s.pinecone_api_key:
        os.environ["PINECONE_API_KEY"] = s.pinecone_api_key
    if s.huggingface_api_token:
        os.environ["HUGGINGFACE_API_TOKEN"] = s.huggingface_api_token
    return s

# Global instance for easy access
settings = get_settings()



