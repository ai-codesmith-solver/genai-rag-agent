"""
test_retriever.py

Verification script to test the end-to-end RAG retrieval pipeline.
This script:
1. Initializes the embedding model.
2. Connects to the existing Pinecone vector store.
3. Performs a similarity search for a sample query.
4. Formats and prints the retrieved context.

How to run:
python -m tests.test_retriever
"""

import logging
from src.config import settings
from src.rag.embeddings import get_embeddings
from src.rag.vector_store import get_existing_vector_store
from src.rag.retriever import create_retriever
from src.helper.utils import get_retrive
from src.tools.tools import search_docs
from dotenv import load_dotenv

load_dotenv() # load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("--- RAG Retrieval Test Started ---")
    
    try:
        # 1. Initialize Embeddings
        logger.info(f"1. Initializing embeddings: {settings.embedding_model_name}...")
        embeddings = get_embeddings(settings.embedding_model_name)
        
        # 2. Access existing Vector Store
        logger.info(f"2. Connecting to Pinecone index: {settings.pinecone_index_name}...")
        vector_store = get_existing_vector_store(settings.pinecone_index_name, embeddings)
        
        # 3. Create Retriever
        logger.info(f"3. Creating retriever (top_k={settings.top_k})...")
        retriever = create_retriever(vector_store, k=settings.top_k)
        
        # 4. Perform Test Query
        query = "What is RAG?"
        logger.info(f"4. Performing test query: '{query}'")
        
        # Perform retrieval
        retrieved_docs = retriever.invoke(query)
        logger.info(f"✓ Retrieved {len(retrieved_docs)} documents.")
        
        # 5. Format and Result using get_retrive
        context = get_retrive(retrieved_docs)
        
        if context:
            print("\n--- Formatted Context Result ---")
            print(context)
            print("---------------------------------\n")
        else:
            logger.warning("No context retrieved. Is the index populated?")

        # Optional: Print retrieved document titles for verification
        titles = search_docs(query, retrieved_docs) #tools function to extract titles from retrieved docs
        logger.info(f"✓ Retrieved document titles: {titles}")

    except Exception as e:
        logger.error(f"An error occurred during retrieval test: {e}", exc_info=True)

if __name__ == "__main__":
    main()

