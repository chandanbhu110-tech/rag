# 📚 RAG Book Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions about its contents.

The application extracts text from the uploaded document, creates vector embeddings, retrieves the most relevant sections, and uses an LLM to generate contextual answers.

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://gfoa97hfnq69qis8ngc2uu.streamlit.app/

## ✨ Features

- 📄 Upload PDF documents
- 🔍 Extract and process document content
- ✂️ Split documents into smaller chunks
- 🧠 Generate embeddings using Hugging Face
- 🗄️ Store embeddings using ChromaDB
- 🔎 Semantic similarity search
- 🤖 Generate answers using Groq LLM
- 💬 Ask natural-language questions about your document
- ⚡ Streamlit-based interactive UI

## 🏗️ RAG Architecture

```text
                PDF Upload
                    │
                    ▼
             Document Loader
                    │
                    ▼
              Text Extraction
                    │
                    ▼
             Text Chunking
                    │
                    ▼
          Hugging Face Embeddings
                    │
                    ▼
               ChromaDB
                    │
                    ▼
              Similarity Search
                    │
                    ▼
             Relevant Chunks
                    │
                    ▼
                Groq LLM
                    │
                    ▼
              AI Generated Answer
