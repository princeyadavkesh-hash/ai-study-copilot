AI Study Copilot
AI-powered PDF question answering system built using Flask, Groq API, and vector-based semantic search.
Users can upload PDFs and ask questions directly from document content with AI-generated answers and citations.

Features


Upload and process PDF documents


AI-powered question answering


Semantic search using vector embeddings


Citation-based answers


Persistent vector storage


Multiple chunk retrieval


Groq LLM integration


Clean responsive UI


Cloud deployment on Render



Tech Stack


Python


Flask


NumPy


Scikit-learn


PDFPlumber


Groq API


HTML


CSS


JavaScript


Render



How It Works


User uploads PDF


PDF text is extracted and chunked


Text chunks are converted into vector embeddings


Relevant chunks are retrieved using semantic similarity


Groq LLM generates answers using retrieved context


Citations are displayed with source references



Project Structure
study_copilot/│├── app.py├── requirements.txt├── Procfile├── runtime.txt│├── core/│   ├── embedder.py│   ├── pdf_processor.py│   ├── vector_store.py│   ├── retriever.py│   └── bm25.py│├── templates/│   └── index.html│├── data/│   ├── uploads/│   └── vector_cache/│└── logs/
Installation
Clone repository:
git clone https://github.com/YOUR_USERNAME/ai-study-copilot.git
Go into project folder:
cd ai-study-copilot
Install dependencies:
pip install -r requirements.txt
Create .env file:
GROQ_API_KEY=your_api_key_here
Run application:
python app.py

Live Demo
https://ai-study-copilot-7ysi.onrender.com

Future Improvements


Chat history memory


Streaming AI responses


Multi-document search


Authentication system


Better UI/UX


Advanced RAG pipeline


LangChain integration


FAISS/ChromaDB support



Author
Prince Yadav