"""
store_index.py

Script to build and populate the Pinecone vector store with embedded document chunks.

This is a setup/initialization script that:
1. Loads markdown documents from a configured data directory.
2. Splits documents into smaller, semantically meaningful chunks.
3. Initializes a HuggingFace embedding model.
4. Creates a Pinecone vector store and populates it with embedded chunks.

Execution flow:
- This script should be run once during project setup or when the knowledge base changes.
- It is not meant to be imported as a library; it performs side effects (index creation).
- All configuration (paths, model names, API keys) is read from src.config.settings.

Dependencies:
- src.rag.doc_loader: Loads and splits documents.
- src.rag.embeddings: Initializes the embedding model.
- src.rag.vector_store: Creates the Pinecone vector store.
- src.config: Provides configuration settings (data_path, embedding_model_name, pinecone_index_name, etc.).

Notes for developers:
- Ensure settings are properly configured before running this script.
- The script will create a new Pinecone index if it does not exist.
- For large document sets, this script may take several minutes to complete.
- Network connectivity to HuggingFace and Pinecone is required.
"""

import logging
from src.config import settings
from src.rag.doc_loader import load_markdown_files, split_documents
from src.rag.embeddings import get_embeddings
from src.rag.vector_store import create_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("--- Vector Store Ingestion Started ---")
    
    try:
        # ============================================================================
        # Step 1: Load markdown documents from the configured data directory.
        # ============================================================================
        logger.info(f"1. Loading markdown files from: {settings.data_path}")
        extracted_data = load_markdown_files(settings.data_path)
        logger.info(f"✓ Loaded {len(extracted_data)} markdown documents.")

        # ============================================================================
        # Step 2: Split documents into smaller, manageable chunks.
        # ============================================================================
        logger.info("2. Splitting documents into chunks...")
        text_chunks = split_documents(extracted_data)
        logger.info(f"✓ Created {len(text_chunks)} text chunks.")

        # ============================================================================
        # Step 3: Initialize the HuggingFace embedding model.
        # ============================================================================
        logger.info(f"3. Initializing embedding model: {settings.embedding_model_name}...")
        embeddings = get_embeddings(settings.embedding_model_name)
        logger.info("✓ Embedding model initialized.")

        # ============================================================================
        # Step 4: Create the Pinecone vector store and populate it.
        # ============================================================================
        logger.info(f"4. Creating/Updating Pinecone index: {settings.pinecone_index_name}...")
        create_vector_store(text_chunks, embeddings, settings.pinecone_index_name)
        logger.info(f"✓ Vector store populated successfully in index: {settings.pinecone_index_name}")

        logger.info("✓ Index setup complete! Knowledge base is ready.")

    except Exception as e:
        logger.error(f"Failed to ingest knowledge base: {e}", exc_info=True)

if __name__ == "__main__":
    main()
