# PDF RAG Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF document and ask questions based strictly on its content.

The system extracts text from the uploaded PDF, splits it into chunks, generates semantic embeddings, stores them in a FAISS vector index, retrieves the most relevant chunks for a question, and uses Google Gemini to generate a grounded answer.

It also includes an evaluation framework for measuring retrieval quality, answer quality, recall, and unanswerable-query handling.

---

## Features

- PDF document upload
- PDF text extraction using PyPDF
- Text cleaning and chunking with overlap
- Semantic embeddings using Sentence Transformers
- FAISS vector similarity search
- Similarity threshold for retrieval
- Grounded answer generation using Google Gemini
- Source and similarity-score display
- Protection against answering questions outside the uploaded document
- FastAPI endpoints for API-based access
- Streamlit web interface
- RAG evaluation framework
- Recall@1, Recall@3, and Recall@5 evaluation
- Retrieval keyword accuracy evaluation
- Answer faithfulness, relevance, and correctness evaluation
- Unanswerable-query rejection evaluation

---

## System Architecture

```text
                 ┌──────────────────┐
                 │    PDF Upload    │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  PDF Extraction  │
                 │     PyPDF        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Text Cleaning &  │
                 │     Chunking     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Embeddings    │
                 │ SentenceTransform│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │   FAISS Index    │
                 │ Vector Retrieval │
                 └────────┬─────────┘
                          │
                    User Question
                          │
                          ▼
                 ┌──────────────────┐
                 │ Query Embedding  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Top-K Retrieval  │
                 │ + Similarity     │
                 │    Threshold     │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Retrieved Context│
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  Google Gemini   │
                 │ Grounded Answer  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Answer + Sources │
                 └──────────────────┘
