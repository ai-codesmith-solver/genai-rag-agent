"""
utils.py

This module contains core utility functions for the RAG (Retrieval-Augmented Generation) 
pipeline. It focuses on administrative tasks like formatting retrieved data 
and extracting clean metadata for user-facing outputs.

Functions:
- get_retrive: Transforms raw LangChain Document objects into a single context string.
- _get_title: Extracts a human-readable title from document metadata.
"""



def get_retrive(retrieved_docs):
    """
    Build a context text from retrieved documents.

    Parameters:
    - retrieved_docs (iterable): Iterable of objects with a `.page_content` attribute.

    Returns:
    - str: Concatenated document contents separated by two newlines.
    """
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


def get_title(doc) -> str:
    """Helper to safely extract document title."""
    source = doc.metadata.get("source", "Unknown")
    return source.split("\\")[-1].split("/")[-1].replace(".md", "")
