from flask import Flask, render_template, request, jsonify
import os
import requests
from dotenv import load_dotenv

from core.vector_store import VectorStore
from core.embedder import LightweightEmbedder
from core.pdf_processor import process_pdf
from core.bm25 import BM25

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================================
# GLOBALS
# =========================================

vector_store = VectorStore()

embedder = LightweightEmbedder()

bm25 = BM25()

all_chunks = []

chat_history = []

# =========================================
# HOME
# =========================================

@app.route("/")
def home():

    return render_template("index.html")

# =========================================
# DOCUMENTS
# =========================================

@app.route("/documents")
def documents():

    docs = []

    unique_docs = {}

    for chunk in all_chunks:

        name = chunk.doc_name

        if name not in unique_docs:
            unique_docs[name] = 0

        unique_docs[name] += 1

    for doc_name, count in unique_docs.items():

        docs.append({
            "name": doc_name,
            "chunks": count
        })

    return jsonify({
        "documents": docs,
        "total_documents": len(docs),
        "total_chunks": len(all_chunks)
    })

# =========================================
# CLEAR
# =========================================

@app.route("/clear", methods=["POST"])
def clear_knowledge_base():

    global vector_store
    global embedder
    global bm25
    global all_chunks
    global chat_history

    vector_store = VectorStore()

    embedder = LightweightEmbedder()

    bm25 = BM25()

    all_chunks = []

    chat_history = []

    return jsonify({
        "message": "Knowledge base cleared successfully."
    })

# =========================================
# UPLOAD PDF
# =========================================

@app.route("/upload", methods=["POST"])
def upload_pdf():

    global vector_store
    global embedder
    global bm25
    global all_chunks

    try:

        if "pdf" not in request.files:

            return jsonify({
                "error": "No PDF uploaded."
            })

        pdf_file = request.files["pdf"]

        if pdf_file.filename == "":

            return jsonify({
                "error": "No file selected."
            })

        clean_name = (
            pdf_file.filename
            .replace(".pdf", "")
            .strip()
        )

        upload_folder = "data/uploads"

        os.makedirs(upload_folder, exist_ok=True)

        save_path = os.path.join(
            upload_folder,
            pdf_file.filename
        )

        pdf_file.save(save_path)

        parsed_doc = process_pdf(save_path)

        new_chunks = parsed_doc.chunks

        # LIMIT FOR FREE TIER
        new_chunks = new_chunks[:40]

        all_chunks.extend(new_chunks)

        corpus = [
            chunk.text
            for chunk in all_chunks
        ]

        # LIGHTWEIGHT EMBEDDING
        embedder.fit(corpus)

        # BM25
        bm25.fit(corpus)

        return jsonify({
            "message": f"""
PDF indexed successfully.

Documents:
1

Total Chunks:
{len(corpus)}
"""
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# =========================================
# ASK
# =========================================

@app.route("/ask", methods=["POST"])
def ask_question():

    global embedder
    global bm25
    global all_chunks

    try:

        data = request.get_json()

        question = data.get(
            "question",
            ""
        ).strip()

        history = data.get(
            "history",
            []
        )

        selected_doc = data.get(
            "selected_doc",
            "All Documents"
        )

        if not question:

            return jsonify({
                "answer": "Please enter a question.",
                "citations": []
            })

        if len(all_chunks) == 0:

            return jsonify({
                "answer": "Please upload a PDF first.",
                "citations": []
            })

        # =========================================
        # TF-IDF SEARCH
        # =========================================

        search_results = embedder.search(
            question,
            top_k=4
        )

        filtered_results = []

        for result in search_results:

            idx = result["index"]

            if idx >= len(all_chunks):
                continue

            chunk = all_chunks[idx]

            if (
                selected_doc != "All Documents"
                and chunk.doc_name != selected_doc
            ):
                continue

            filtered_results.append({
                "chunk": chunk,
                "score": result["score"]
            })

        if len(filtered_results) == 0:

            return jsonify({
                "answer": "I could not find that information in the uploaded PDFs.",
                "citations": []
            })

        # =========================================
        # CONTEXT
        # =========================================

        context = "\n\n".join([
            item["chunk"].text
            for item in filtered_results
        ])

        # =========================================
        # HISTORY
        # =========================================

        history_text = ""

        recent_history = history[-6:]

        for msg in recent_history:

            role = msg.get("role", "")

            content = msg.get("content", "")

            history_text += f"{role}: {content}\n"

        # =========================================
        # PROMPT
        # =========================================

        prompt = f"""
You are an AI Study Copilot.

Answer ONLY using the PDF context.

If answer is missing, say:
"I could not find that information in the uploaded PDFs."

====================
Conversation
====================

{history_text}

====================
PDF Context
====================

{context}

====================
Question
====================

{question}
"""

        # =========================================
        # GROQ
        # =========================================

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        result = response.json()

        answer = result["choices"][0]["message"]["content"]

        citations = []

        for rank, item in enumerate(
            filtered_results,
            start=1
        ):

            chunk = item["chunk"]

            citations.append({
                "source": chunk.doc_name,
                "page": chunk.page_num,
                "rank": rank,
                "score": round(item["score"], 3),
                "preview": chunk.preview
            })

        return jsonify({
            "answer": answer,
            "citations": citations
        })

    except Exception as e:

        return jsonify({
            "answer": str(e),
            "citations": []
        })

# =========================================
# RUN
# =========================================

if __name__ == "__main__":

    app.run(debug=True)