"""
test_agent.py

Verification script to test the end-to-end AI agent orchestration.
This script:
1. Initializes the agent.
2. Detected intent and agent decisions.
3. Performs a grounded Q&A using the RAG pipeline.
4. Logs the entire process with timestamps.

How to run:
python -m tests.test_agent
"""

import logging
from src.agent.agent import run_agent
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Configure logging to console
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("--- AI Agent Orchestration Test Started ---")
    
    # Sample queries to test different intents
    test_queries = [
        "What is RAG?", # Expected: Explanation
        "What is the difference between RAG and AI Agents?", # Expected: Comparison
        "How do I use FastAPI?" # Expected: Explanation
    ]

    try:
        for query in test_queries:
            logger.info(f"\n>>> Running Agent for Query: '{query}'")
            
            # Run the agent
            result = run_agent(query)
            
            # Log agent decisions and metadata
            logger.info(f"Agent Decision: {result['agent_decision']}")
            logger.info(f"Documents Used: {result['documents_used']}")
            
            # Print the final answer clearly
            print(f"\nANSWER:\n{result['answer']}")
            print("-" * 50)

    except Exception as e:
        logger.error(f"An error occurred during agent test: {e}", exc_info=True)
        
    logger.info("--- AI Agent Orchestration Test Finished ---")

if __name__ == "__main__":
    main()
