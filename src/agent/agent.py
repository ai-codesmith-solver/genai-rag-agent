"""
agent.py

The 'Brain' of the RAG system. This module handles explicit decision logic 
to determine the user's intent and decide how the retrieved information 
should be processed before being sent to the LLM.

Key Logic:
- Intent Detection: Checks for comparison keywords to decide between 'comparison' and 'explanation' modes.
- Context Filtering: Decides whether to use all retrieved documents or just the top result based on intent.
"""

from src.generate_answer import generate_answer
from typing import Dict
from src.retrieve_relevant_docs import retrieve_relevant_docs
from src.tools.tools import search_docs
from src.helper.utils import get_title

def run_agent(query: str) -> Dict:
    """
    Main orchestration function for the AI agent.
    
    Workflow:
    1. Retrieves relevant documents from the vector store.
    2. Analyzes the query to detect 'intent' (e.g., comparison vs explanation).
    3. Selects the appropriate documents based on the detected intent.
    4. Generates a grounded answer using the LLM.
    
    Returns: A dictionary containing the answer, source titles, and agent decision.
    """

    retrieved_docs = retrieve_relevant_docs(query)

    if not retrieved_docs:
        return {
            "answer": "I don’t have enough information in my knowledge base.",
            "documents_used": [],
            "agent_decision": "insufficient_context"
        }
    
    query_lower = query.lower()
    
    # Define comparison keywords
    comparison_keywords = ["compare", "difference", "vs", "versus"]
    is_comparison = any(kw in query_lower for kw in comparison_keywords)

    # Decision Logic: Determine intent and select documents
    if is_comparison:
        intent = "comparison"
        agent_decision = "combined_multiple_docs"
        selected_titles = search_docs(query, retrieved_docs)
        docs_to_use = [
            doc for doc in retrieved_docs
            if get_title(doc) in selected_titles
        ]
    else:
        # Default fallback is 'explanation'
        intent = "explanation"
        agent_decision = "answered_using_retrieved_context"
        selected_titles = [get_title(retrieved_docs[0])]
        docs_to_use = [retrieved_docs[0]]

    # Generate answer using the orchestrated state
    answer = generate_answer(
        query=query,
        retrieved_docs=docs_to_use,
        intent=intent,
        agent_decision=agent_decision
    )

    return {
        "answer": answer,
        "documents_used": selected_titles,
        "agent_decision": agent_decision
    }
