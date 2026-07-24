# RAG PDF Assistant

This project is a simple RAG (Retrieval-Augmented Generation) application that lets users upload a PDF and ask questions about its content. It uses LangChain for document processing, ChromaDB for storing embeddings, HuggingFace for generating embeddings, and Mistral AI to answer questions based on the uploaded document.

## Features

- Upload any PDF file
- Create a vector database from the PDF
- Ask questions about the document
- View the retrieved document chunks
- Chat interface built with Streamlit
- Uses HuggingFace embeddings instead of OpenAI embeddings

## Technologies Used

- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Mistral AI
- PyPDF
- python-dotenv

## Project Structure

RAG Project
├─ app.py
├─ main.py
├─ create_database.py
├─ requirements.txt
├─ .env
├─ chroma_db/
├─ uploads/
└─ document loaders

## How It Works

1. Upload a PDF.
2. The PDF is split into smaller chunks.
3. HuggingFace creates embeddings for each chunk.
4. ChromaDB stores the embeddings.
5. When a question is asked, the most relevant chunks are retrieved.
6. Mistral AI generates an answer using only the retrieved content.

## Future Improvements

- Support multiple PDF uploads
- Display page numbers in responses
- Add conversation memory
- Improve the user interface
- Add support for more document formats
