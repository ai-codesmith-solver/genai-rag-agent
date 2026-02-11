"""
embeddings.py

Small wrapper to create a HuggingFace embeddings client.

Provides:
- get_embeddings(model_name: str) -> HuggingFaceEndpointEmbeddings

Notes:
- model_name should match the HuggingFace endpoint/model identifier expected by
  HuggingFaceEndpointEmbeddings.
- This module returns the embeddings client object (it does not compute vectors).
"""

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from src.config import settings

def get_embeddings(model_name:str):
    """
    Initialize and return a HuggingFaceEndpointEmbeddings instance.

    Parameters:
    - model_name (str): Model or endpoint identifier to use for embeddings.

    Returns:
    - HuggingFaceEndpointEmbeddings: Client instance ready to produce embeddings.

    Example:
        emb_client = get_embeddings("all-MiniLM-L6-v2")
        vectors = emb_client.embed_documents(["example text"])
    """
    embeddings=HuggingFaceEndpointEmbeddings(model=model_name, huggingfacehub_api_token=settings.huggingface_api_token)
    return embeddings