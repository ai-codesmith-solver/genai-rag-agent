"""
tools.py

This module contains the functional 'tools' used by the AI agent.
Specifically, it provides logic to filter and verify relevant documents
after the initial vector search.
"""

from typing import List, Dict

def search_docs(query: str, retrieve_documents: List[Dict]) -> List[str]:
    """
    Selects and returns the titles of documents from the retrieved set that 
    specifically contain keywords from the user's query.

    This acts as a second-pass filter to ensure high relevance and to 
    provide the user with a clear list of 'Documents Used'.

    Parameters:
        query (str): The raw text of the user's question.
        retrieve_documents (List[Dict]): A list of LangChain Document objects 
                                   returned by the vector store search.

    Returns:
        List[str]: A list of clean document titles (filenames).
    """

    # 1. Prepare query words for matching
    query_words = set(query.lower().split()) 
    selected_titles = []

    # 2. Iterate through documents to extract titles and verify relevance
    for doc in retrieve_documents:
        # Extract title from 'source' metadata (handling both Windows and Unix paths)
        # Example: "data\rag_overview.md" -> "rag_overview"
        source = doc.metadata.get("source", "Unknown")
        title = source.split("\\")[-1].split("/")[-1].replace(".md", "")
        
        # Combine title and content for broader keyword matching
        content = doc.page_content.lower()
        combined_text = (title + " " + content).lower()

        # 3. Simple keyword relevance check
        # If any word from the query is found in the title or content, we include it.
        for word in query_words:
            if word in combined_text:
                selected_titles.append(title)
                break # Move to next document after first keyword match

    # 4. Fallback Logic:
    # If no documents matched the specific keywords (e.g., query uses synonyms),
    # we return the titles of all retrieved documents as a baseline.
    if not selected_titles:
        for doc in retrieve_documents:
            source = doc.metadata.get("source", "Unknown")
            title = source.split("\\")[-1].split("/")[-1].replace(".md", "")
            selected_titles.append(title)

    return selected_titles

