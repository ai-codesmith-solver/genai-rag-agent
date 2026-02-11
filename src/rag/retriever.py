"""
retriever.py

Utilities to create and configure retriever objects from vector stores.

This module provides:
- create_retriever(vector_store, k=4) -> Retriever: Converts a vector store into a retriever
  that performs similarity-based document retrieval.

Key considerations for developers:
- The retriever uses the underlying vector store's similarity search mechanism.
- The 'k' parameter controls how many top-ranked similar documents are returned.
- Retrievers are typically used in RAG (Retrieval-Augmented Generation) pipelines to fetch
  relevant context before passing it to a language model.
- Different vector stores (Pinecone, FAISS, Chroma, etc.) can be converted to retrievers
  using this same pattern.
"""


def create_retriever(vector_store, k:int=4):
    """
    Create and return a retriever object for the given vector store.

    Parameters:
    - vector_store: Vector store instance (e.g., PineconeVectorStore) to use for retrieval.
    - k (int): Number of top similar documents to retrieve (default: 4).

    Returns:
    - Retriever: A retriever object configured to query the provided vector store.

    Notes:
    - The retriever will use the specified vector store to perform similarity search.
    - Adjust 'k' based on the desired number of results and application needs.
    """
    retriever=vector_store.as_retriever(search_kwargs={"k": k})
    return retriever