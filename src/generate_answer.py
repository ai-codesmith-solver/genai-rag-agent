"""
generate_answer.py

Orchestrates the final stage of the RAG pipeline: generating a response 
using an LLM. It uses LangChain Expression Language (LCEL) to create a 
declarative chain that combines context, prompt, and model.

Core Pipeline:
RunnableParallel -> PromptTemplate -> ChatModel -> StrOutputParser
"""

from typing import List, Dict
from src.prompt import prompt
from src.llm import get_llm
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.helper.utils import get_retrive

# Shared output parser to convert LLM messages to clean strings
str_parse = StrOutputParser()

# Shared LLM instance
llm = get_llm()

def generate_answer(query: str, retrieved_docs: List[Dict], intent: str, agent_decision: str) -> str:
    """
    Main entry point for generating a grounded answer.
    
    Args:
        query: The user's question.
        retrieved_docs: List of documents selected by the agent.
        intent: The detected intent (explanation/comparison).
        agent_decision: The explicit decision made by the agent.
        
    Returns:
        A formatted string answer from the LLM.
    """

    context = get_retrive(retrieved_docs)

    parallel_chain=RunnableParallel({
        'context': RunnableLambda(lambda x: context),
        'query': RunnablePassthrough(),
        'agent_decision':RunnableLambda(lambda x: agent_decision),
        'intent': RunnableLambda(lambda x: intent)
    })

    rag_chain= parallel_chain | prompt | llm | str_parse

    result=rag_chain.invoke(query)
    return result
