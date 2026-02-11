# RAG Overview

Retrieval-Augmented Generation (RAG) is a sophisticated architecture designed to optimize the output of Large Language Models (LLMs) by referencing an authoritative, external knowledge base outside of its initial training data.

## Core Benefits
- **Factuality**: Reduces hallucinations by forcing the model to cite specific context.
- **Cost-Efficiency**: It is significantly cheaper to update a vector database than it is to fine-tune a model.
- **Recency**: Allows the AI to answer questions about events or data that occurred after its training cutoff.

## The RAG Lifecycle
1. **Retrieval**: The system searches for relevant documents based on the user's query using semantic search.
2. **Augmentation**: The retrieved information is appended to the original user prompt.
3. **Generation**: The LLM uses the combined prompt (Question + Context) to produce a grounded response.
