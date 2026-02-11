"""
app.py

Main entry point for the GenAI RAG & Agent FastAPI application.
This module defines the external API interface, handles request validation, 
and orchestrates the agent's query process.
"""

from fastapi import FastAPI, HTTPException
from src.schemas import AgentQueryRequest, AgentQueryResponse
from src.agent.agent import run_agent
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

app = FastAPI(title="GenAI RAG & Agent API")

@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(request: AgentQueryRequest):
    """
    Endpoint to process user queries through the AI agent.
    
    This function:
    1. Validates that the query is not empty.
    2. Passes the query to the agent's 'Brain' (run_agent).
    3. Returns the agent's grounded response along with metadata.
    """


    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    response = run_agent(request.query)

    return response
