# FastAPI Basics

FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints. It is one of the fastest Python frameworks available, rivaling NodeJS and Go in terms of speed.

## Key Features
- **Speed**: Extreme performance thanks to Starlette and Pydantic.
- **Auto-Documentation**: Automatically generates interactive API documentation (Swagger UI and ReDoc).
- **Type Safety**: Leverages Python type hints for rigorous data validation and editor support.
- **Asynchronous Flow**: Native support for `async` and `await`, which is ideal for I/O bound tasks like calling LLM APIs.

## Why use FastAPI for AI?
AI applications often involve long-running LLM requests. FastAPI's asynchronous nature allows the server to handle many concurrent connections without blocking, making it the industry standard for GenAI backends.
