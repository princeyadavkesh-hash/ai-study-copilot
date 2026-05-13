# AI Study Copilot

AI Study Copilot is a lightweight RAG (Retrieval-Augmented Generation) application that allows users to upload PDF documents and ask questions directly from them using AI.

The project uses:
- Flask backend
- PDF processing
- Semantic search
- Vector embeddings
- BM25 retrieval
- Groq LLM API
- Hybrid search pipeline

---

# Live Demo

🚀 Live Project:

https://ai-study-copilot-7ysi.onrender.com

---

# Features

- Upload PDF documents
- Extract and chunk PDF text
- Semantic vector search
- BM25 keyword retrieval
- Hybrid retrieval pipeline
- AI-generated answers
- Source citations
- Multi-document support
- Persistent vector storage
- Clean dark UI

---

# Tech Stack

## Backend
- Python
- Flask

## AI / NLP
- Groq API
- BM25 Retrieval
- TF-IDF Vectorization
- Cosine Similarity

## PDF Processing
- pdfplumber

## Deployment
- Render

---

# Project Structure

```text
study_copilot/
│
├── core/
│   ├── bm25.py
│   ├── embedder.py
│   ├── pdf_processor.py
│   ├── retriever.py
│   └── vector_store.py
│
├── templates/
│   └── index.html
│
├── data/
│   ├── uploads/
│   └── vector_cache/
│
├── logs/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── Procfile
└── README.md
```

---

# How It Works

1. User uploads PDF
2. PDF text gets extracted
3. Text is divided into chunks
4. Chunks are converted into embeddings
5. Vector store indexes embeddings
6. User asks question
7. Hybrid retrieval finds best chunks
8. Groq LLM generates final answer
9. Citations are displayed

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/study_copilot.git

cd study_copilot
```

---

# Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

## Windows

```bash
venv\Scripts\activate
```

## Linux / Mac

```bash
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

# Run Project

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# Deployment

This project is deployed on Render.

Files used for deployment:
- Procfile
- runtime.txt
- requirements.txt

---

# Future Improvements

- Chat history memory
- Streaming AI responses
- Authentication system
- Advanced RAG pipeline
- Better UI/UX
- Database integration
- OCR support
- Notes generation
- Quiz generation

---

# Screenshots

## Home Page

![Home](assets/home.png)

---

## Upload PDF

![Upload](assets/upload.png)

---

## AI Answer

![Answer](assets/answer.png)

---

# Resume Project Description

AI-powered PDF Question Answering system using Retrieval-Augmented Generation (RAG), semantic search, BM25 retrieval, vector embeddings, and Groq LLM API. Built with Flask and deployed on Render.

---

# Author

Prince Yadav
B.Tech AI Engineering Student

GitHub:
https://github.com/princeyadavkesh-hash