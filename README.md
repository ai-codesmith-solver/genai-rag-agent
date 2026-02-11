# GenAI RAG Agent: Knowledge-Based Assistant

A modular, industry-grade Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain**, and **Google Gemini**. This agent provides grounded answers based strictly on a localized knowledge base.

## 🚀 How to Run

### 1. Prerequisities
- Python 3.10+
- A Google Gemini API Key
- A Pinecone API Key (Serverless index)

### 2. Installation
Clone the repository and install dependencies:
```powershell
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory and add your credentials:
```env
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
HUGGINGFACE_API_TOKEN=your_hf_token
```

### 4. Ingest Data (Knowledge Base)
Populate your Pinecone vector store with the documents in the `data/` folder:
```powershell
python -m src.helper.store_index
```

### 5. Start the API
Run the FastAPI server:
```powershell
uvicorn app:app --reload
```

### 6. Run Tests
You can verify the system using the built-in test scripts:
```powershell
# Test Retrieval only
python -m tests.test_retriever

# Test full Agent Orchestration
python -m tests.test_agent
```

---

## 🧠 How the Agent Works

The agent is designed as a **Modular Decision Engine** rather than a black-box LLM call.

1.  **Intent Detection**: The "Brain" (`src/agent/agent.py`) first analyzes the user query. It looks for "Comparison" keywords (vs, difference, compare).
2.  **Strategic Retrieval**:
    -   **Explanation Intent**: Retrieves and uses only the top-ranked document for high precision.
    -   **Comparison Intent**: Uses multiple retrieved documents to provide a balanced overview.
3.  **Grounded Prompting**: The agent uses a strict System Prompt (`src/prompt.py`) that forbids hallucinations and forces use of the `{context}` variable.
4.  **Verification Tool**: A secondary `search_docs` tool confirms the relevance of retrieved titles before the final answer is generated.

---

## 📝 Assumptions Made

-   **Document Format**: The system assumes knowledge is stored as `.md` files in the `data/` directory.
-   **Embedding Model**: Uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) for local, fast embedding generation.
-   **Vector Metric**: The Pinecone index is assumed to use **Cosine Similarity** for optimal semantic matching.
-   **Strict Grounding**: It is assumed that if information is missing from the knowledge base, the agent **must** refuse to answer (preventing external knowledge leakage).
-   **Clean Path Handling**: The system assumes a mix of Windows/Unix development environments and robustly handles file paths in metadata extraction.
