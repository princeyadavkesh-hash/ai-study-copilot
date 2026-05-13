from flask import Flask, render_template, request, jsonify
import os
import requests
from dotenv import load_dotenv

from core.vector_store import VectorStore
from core.pdf_processor import process_pdf

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================================
# VECTOR STORE
# =========================================

loaded_store = VectorStore.load()

if loaded_store is not None:

    vector_store = loaded_store

    print("Loaded persisted vector store.")

else:

    vector_store = VectorStore()

    print("Started fresh vector store.")

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

    for doc_name in vector_store.document_names:

        chunk_count = len([
            chunk
            for chunk in vector_store.chunks
            if chunk.doc_name == doc_name
        ])

        docs.append({
            "name": doc_name,
            "chunks": chunk_count
        })

    return jsonify({
        "documents": docs,
        "total_documents": len(docs),
        "total_chunks": len(vector_store.chunks)
    })

# =========================================
# CLEAR
# =========================================

@app.route("/clear", methods=["POST"])
def clear_knowledge_base():

    global vector_store

    vector_store.reset()

    vector_store = VectorStore()

    return jsonify({
        "message": "Knowledge base cleared successfully."
    })

# =========================================
# UPLOAD PDF
# =========================================

@app.route("/upload", methods=["POST"])
def upload_pdf():

    global vector_store

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

        if vector_store.has_document(clean_name):

            return jsonify({
                "message": "Document already indexed."
            })

        upload_folder = "data/uploads"

        os.makedirs(upload_folder, exist_ok=True)

        save_path = os.path.join(
            upload_folder,
            pdf_file.filename
        )

        pdf_file.save(save_path)

        parsed_doc = process_pdf(
            save_path,
            chunk_index_offset=len(vector_store.chunks)
        )

        chunks = parsed_doc.chunks

        embeddings = []

        for chunk in chunks:

            # SUPER LIGHTWEIGHT EMBEDDING
            embeddings.append([
                len(chunk.text),
                chunk.page_num,
                chunk.chunk_index
            ])

        vector_store.add_embeddings(
            embeddings,
            chunks
        )

        vector_store.save()

        return jsonify({
            "message": f"""
PDF indexed successfully.

Documents:
{len(vector_store.document_names)}

Total Chunks:
{len(vector_store.chunks)}
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

    global vector_store

    try:

        data = request.get_json()

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "answer": "Please enter a question.",
                "citations": []
            })

        if len(vector_store.chunks) == 0:

            return jsonify({
                "answer": "Please upload a PDF first.",
                "citations": []
            })

        # =========================================
        # SIMPLE KEYWORD SEARCH
        # =========================================

        matches = []

        question_lower = question.lower()

        for chunk in vector_store.chunks:

            text_lower = chunk.text.lower()

            score = 0

            for word in question_lower.split():

                if word in text_lower:

                    score += 1

            if score > 0:

                matches.append(
                    (score, chunk)
                )

        matches.sort(
            key=lambda x: x[0],
            reverse=True
        )

        top_chunks = matches[:3]

        if len(top_chunks) == 0:

            return jsonify({
                "answer": "I could not find that information in the uploaded PDFs.",
                "citations": []
            })

        # =========================================
        # CONTEXT
        # =========================================

        context = "\n\n".join([
            chunk.text
            for _, chunk in top_chunks
        ])

        # =========================================
        # PROMPT
        # =========================================

        prompt = f"""
Answer using ONLY this PDF context.

Context:
{context}

Question:
{question}
"""

        # =========================================
        # GROQ API
        # =========================================

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.3-70b-versatile",
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
            json=payload
        )

        print("STATUS:", response.status_code)

        print("RAW RESPONSE:", response.text)

        result = response.json()

        answer = result["choices"][0]["message"]["content"]

        # =========================================
        # CITATIONS
        # =========================================

        citations = []

        for i, (_, chunk) in enumerate(top_chunks):

            citations.append({
                "source": chunk.doc_name,
                "page": chunk.page_num,
                "preview": chunk.preview
            })

        return jsonify({
            "answer": answer,
            "citations": citations
        })

    except Exception as e:

        print("ASK ERROR:", str(e))

        return jsonify({
            "answer": str(e),
            "citations": []
        })

# =========================================
# RUN
# =========================================

if __name__ == "__main__":

    app.run(debug=True)