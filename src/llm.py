"""
llm.py

Utilities to initialize and configure a Google Generative AI (Gemini) language model client.

This module provides:
- get_llm() -> ChatGoogleGenerativeAI: Creates and returns a configured LLM instance
  for text generation tasks in RAG pipelines and agents.

Key considerations for developers:
- The LLM client requires valid Google API credentials (gemini_api_key).
- Temperature controls output randomness: 0.0 = deterministic, 1.0+ = more creative.
- max_output_tokens limits the length of generated responses (consider context window limits).
- This function initializes the client but does not make network calls until inference time.
- The model_name must correspond to a supported Google Gemini model identifier.

Notes:
- Store API key securely in environment variables; do not hardcode in source.
- Different models may have different token limits and capabilities.
- For production use, consider implementing retry logic and error handling at call sites.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from .config import settings

def get_llm():
    """
    Initialize and return a ChatGoogleGenerativeAI instance configured with settings.

    This function creates an LLM client for interacting with Google's Gemini models.
    It uses the API key and model name specified in the settings module.

    Returns:
    - ChatGoogleGenerativeAI: Configured LLM client instance ready for generating text.

    Notes:
    - Ensure that settings.gemini_api_key is set to a valid API key before calling.
    - The model_name in settings should correspond to a supported Gemini model (e.g., "gemini-2.5-flash").
    - This function does not perform any network calls; it only initializes the client.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.model_name, 
        api_key=settings.gemini_api_key, 
        temperature=settings.temperature, 
        max_output_tokens=settings.max_output_tokens
    )
    return llm
