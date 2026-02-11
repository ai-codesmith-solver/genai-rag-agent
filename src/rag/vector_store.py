"""
vector_store.py

Pinecone vector store initialization and management utilities.

This module provides functions to:
- Initialize a Pinecone client with API credentials.
- Create or retrieve Pinecone indexes with specified configurations.
- Create a vector store from document chunks and embeddings.
- Access an existing vector store by index name.

Key considerations for developers:
- Pinecone API key should be passed as a parameter (avoid hardcoding).
- Dimension and metric parameters must match your embedding model's output.
- ServerlessSpec is optional; use it to configure serverless deployment settings.
- Error handling: functions raise ValueError if indexes don't exist or credentials are invalid.
"""

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from ..config import settings

def init_pinecone(pinecone_api_key: str) -> Pinecone:
    """
    Initialize and return a Pinecone client instance.

    Parameters:
    - pinecone_api_key (str): API key for authentication with Pinecone service.

    Returns:
    - Pinecone: Client instance ready to interact with Pinecone indexes.

    Raises:
    - Authentication errors if the API key is invalid or missing.

    Notes:
    - Store API key securely (e.g., environment variable) and do not hardcode in source.
    """
    pc=Pinecone(api_key=pinecone_api_key)
    return pc

def create_pinecone_index(pc:Pinecone, index_name:str, dimension:int, metric:str="cosine", serverless_spec:ServerlessSpec=None) -> None:
    """
    Create a Pinecone index if it does not already exist.

    Parameters:
    - pc (Pinecone): Pinecone client instance (from init_pinecone).
    - index_name (str): Name of the index to create or check for.
    - dimension (int): Vector dimension (must match embedding model output).
    - metric (str): Distance metric for similarity search (default: "cosine").
    - serverless_spec (ServerlessSpec): Optional serverless configuration for the index.

    Returns:
    - None

    Notes:
    - This function checks if the index already exists before creating.
    - If the index exists, no action is taken (safe to call repeatedly).
    - dimension must match the embedding model's output dimension.
    - Common metrics: "cosine", "euclidean", "dotproduct".
    """
    # Correctly check for index existence by names
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        try:
            pc.create_index(
                name=index_name, 
                dimension=dimension, 
                metric=metric, 
                spec=serverless_spec
            )
            print(f"✓ Creating new Pinecone index: {index_name}")
        except Exception as e:
            # Handle the case where index might have been created simultaneously
            if "ALREADY_EXISTS" in str(e):
                print(f"ℹ Index {index_name} already exists (handled via exception).")
            else:
                raise e
    else:
        print(f"ℹ Use existing Pinecone index: {index_name}")


    index=pc.Index(index_name)


def create_vector_store(text_chunks:list, embeddings, index_name:str):
    """
    Create a new vector store from document chunks and embeddings.

    Parameters:
    - text_chunks (list): List of text chunks (LangChain Document objects) to embed and store.
    - embeddings: Embedding model/client instance (e.g., HuggingFaceEndpointEmbeddings).
    - index_name (str): Name of the Pinecone index to create/use.

    Returns:
    - PineconeVectorStore: Vector store instance ready for similarity searches.

    Notes:
    - This function initializes Pinecone, creates the index, and populates it with vectors.
    - Currently passes None for pinecone_api_key and dimension — update to pass actual values.
    - The embedding model will compute vectors for all text_chunks before storing.
    """
    # Initialize Pinecone and ensure index exists
    pc = init_pinecone(settings.pinecone_api_key)
    create_pinecone_index(
        pc=pc, 
        index_name=index_name, 
        dimension=settings.vector_dimension, 
        metric="cosine", 
        serverless_spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    vector_store=PineconeVectorStore.from_documents(
        documents=text_chunks, 
        embedding=embeddings, 
        index_name=index_name,
        pinecone_api_key=settings.pinecone_api_key
    )


    return vector_store


def get_existing_vector_store(index_name:str, embeddings):
    """
    Retrieve an existing vector store by index name.

    Parameters:
    - index_name (str): Name of the existing Pinecone index.
    - embeddings: Embedding model/client instance (must match the one used when creating the store).

    Returns:
    - PineconeVectorStore: Vector store instance connected to the existing Pinecone index.

    Raises:
    - ValueError: If the specified index does not exist in Pinecone.

    Notes:
    - This function does not create or modify the index; it only creates a client reference.
    - The embeddings parameter is required for performing similarity searches on stored vectors.
    - Currently passes None for pinecone_api_key — update to pass actual API key.
    """
    
    vector_store=PineconeVectorStore(
        embedding=embeddings, 
        index_name=index_name,
        pinecone_api_key=settings.pinecone_api_key
    )

    return vector_store