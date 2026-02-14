# 📚 Mini NotebookLM - RAG Based PDF Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant that allows users to upload a PDF and ask contextual questions.

## 🚀 Features
- PDF upload via Streamlit
- Text extraction using PyPDF
- Semantic search using Sentence Transformers
- Vector similarity search using FAISS
- Context-aware question answering using FLAN-T5

## 🧠 Architecture
PDF → Text Extraction → Chunking → Embeddings → FAISS → Retrieval → LLM Answer

## 🛠️ Tech Stack
- Python
- Streamlit
- Hugging Face Transformers
- Sentence Transformers
- FAISS

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
