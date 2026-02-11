# Vector Databases

Vector databases are specialized storage systems designed to handle "embeddings"—mathematical representations of data that capture semantic meaning.

## How it Works
Unlike traditional relational databases that use exact matches, vector databases use proximity. They convert text into multi-dimensional vectors (dense arrays of numbers) and find the "closest" matches based on the meaning of the words.

## Distance Metrics
- **Cosine Similarity**: Measures the angle between two vectors (standard for text RAG).
- **Euclidean Distance**: Measures the straight-line distance between points.
- **Dot Product**: Measures how much two vectors point in the same direction.

## Popular Providers
Industry leaders include **Pinecone** (managed/serverless), **ChromaDB** (local/open-source), and **Weaviate**.
