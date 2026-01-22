# RAG PDF Chatbot (Streamlit + FAISS + Local Embeddings + Ollama)

This project is a PDF Question Answering chatbot built using the RAG (Retrieval-Augmented Generation) approach. You can upload any PDF, ask questions, and get answers based on the document content with source references.

It uses FAISS for vector search, HuggingFace embeddings for indexing the PDF locally (free), and Ollama for running the LLM locally (no OpenAI API key required).

## Features
- Upload a PDF and ask questions
- Semantic search using FAISS
- Local embeddings (free, offline)
- Local LLM answering using Ollama (no API key required)
- Displays source chunks used for the answer

## Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- PyMuPDF
- HuggingFace embeddings
- Ollama


## Prerequisites
- Python 3.10 or above

Check your Python version:
```bash
python --version
Author

Srujan Kisara
