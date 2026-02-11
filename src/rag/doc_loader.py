"""
doc_loader.py

Utilities to load markdown files from a directory into LangChain Document objects
and to split those documents into smaller chunks suitable for embedding or indexing.

This module provides:
- load_markdown_files(data: str) -> list: Loads all .md files under `data` using
  UnstructuredMarkdownLoader via DirectoryLoader.
- split_documents(documents, chunk_size=1000, chunk_overlap=200) -> list:
  Splits a list of Documents into smaller chunks using RecursiveCharacterTextSplitter.

Notes for other developers:
- The functions return LangChain Document objects (not plain strings).
- chunk_size and chunk_overlap are character counts; adjust based on your model/tokenizer.
- DirectoryLoader.glob currently set to "*.md" — change if you need other file types.
"""

from langchain_community.document_loaders import UnstructuredMarkdownLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_markdown_files(data:str):
    """
    Load all markdown files from a directory.
    - loader_cls: injectable loader class (improves testability and OCP).
    - glob: file pattern to load.
    """
    loader=DirectoryLoader(data, glob="*.md", loader_cls=UnstructuredMarkdownLoader)
    documents=loader.load()
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks.
    - splitter: optional injected splitter instance (supports custom splitting strategies).
    """
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks
