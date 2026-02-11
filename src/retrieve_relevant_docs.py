"""
retrieve_relevant_docs.py

Encapsulates the retrieval logic for the RAG pipeline. This module abstracts 
the details of embeddings, Pinecone connections, and LangChain retrievers 
to provide a simple interface for the agent to fetch relevant context.
"""

from src.rag.vector_store import get_existing_vector_store
from src.rag.embeddings import get_embeddings
from typing import Dict, List
from src.rag.retriever import create_retriever
from src.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def retrieve_relevant_docs(query: str) -> List[Dict]:
    """
    Connects to the vector store and retrieves documents relevant to the query.
    
    Workflow:
    1. Loads the embedding model specified in settings.
    2. Connects to the existing Pinecone index.
    3. Creates a retriever with the configured Top-K value.
    4. Invokes the retriever to get relevant documents.
    """

    # 1. Initialize Embeddings
    logger.info(f"1. Initializing embeddings: {settings.embedding_model_name}...")
    embeddings = get_embeddings(settings.embedding_model_name)
        
    # 2. Access existing Vector Store
    logger.info(f"2. Connecting to Pinecone index: {settings.pinecone_index_name}...")
    vector_store = get_existing_vector_store(settings.pinecone_index_name, embeddings)
        
    # 3. Create Retriever
    logger.info(f"3. Creating retriever (top_k={settings.top_k})...")
    retriever = create_retriever(vector_store, k=settings.top_k)
        
    # Perform retrieval
    retrieved_docs = retriever.invoke(query)
    logger.info(f"✓ Retrieved {len(retrieved_docs)} documents.")

    return retrieved_docs