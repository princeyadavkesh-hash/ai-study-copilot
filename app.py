from flask import Flask, render_template, request, jsonify, Response
import os
import json
import requests
from dotenv import load_dotenv

from core.vector_store import VectorStore
from core.embedder import LSAEmbedder
from core.pdf_processor import process_pdf
from core.bm25 import BM25

load_dotenv()

app = Flask(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================================
# LOAD VECTOR STORE
# =========================================

loaded_store = VectorStore.load()

if loaded_store is not None:

    vector_store = loaded_store

    all_chunks = loaded_store._chunks

    print("Loaded persisted vector store.")

else:

    vector_store = VectorStore()

    all_chunks = []

    print("Started fresh vector store.")

# =========================================
# GLOBALS
# =========================================

embedder = LSAEmbedder()

bm25 = BM25()

chat_history = []

# =========================================
# REBUILD BM25 + EMBEDDINGS
# =========================================

if len(all_chunks) > 0:

    corpus = [
        chunk.text
        for chunk in all_chunks
    ]

    embedder.fit(corpus)

    bm25.fit(corpus)

    print(f"Restored {len(all_chunks)} chunks.")

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
            c for c in all_chunks
            if c.doc_name == doc_name
        ])

        docs.append({
            "name": doc_name,
            "chunks": chunk_count
        })

    return jsonify({
        "documents": docs,
        "total_documents": len(docs),
        "total_chunks": len(all_chunks)
    })

# =========================================
# RESET
# =========================================

@app.route("/reset", methods=["POST"])
def reset():

    global vector_store
    global embedder
    global bm25
    global all_chunks
    global chat_history

    vector_store.reset()

    vector_store = VectorStore()

    embedder = LSAEmbedder()

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

        document_name = os.path.splitext(
            pdf_file.filename
        )[0]

        if vector_store.has_document(
            document_name
        ):

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
            doc_name=document_name,
            chunk_index_offset=len(all_chunks)
        )

        new_chunks = parsed_doc.chunks

        all_chunks.extend(new_chunks)

        corpus = [
            chunk.text
            for chunk in all_chunks
        ]

        embedder.fit(corpus)

        embeddings = embedder.transform(corpus)

        vector_store = VectorStore()

        vector_store.add_embeddings(
            embeddings,
            all_chunks
        )

        vector_store.save()

        bm25.fit(corpus)

        return jsonify({
            "message": f"""
PDF indexed successfully.

Documents:
{len(vector_store.document_names)}

Total Chunks:
{len(all_chunks)}
"""
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

# =========================================
# STREAMING ASK
# =========================================

@app.route("/ask", methods=["POST"])
def ask_question():

    global vector_store
    global embedder
    global bm25
    global all_chunks
    global chat_history

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

        selected_document = data.get(
            "selected_document",
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
        # SEARCH
        # =========================================

        query_embedding = embedder.transform_one(
            question
        )

        semantic_results = vector_store.search(
            query_embedding,
            top_k=10
        )

        # =========================================
        # FILTER DOCUMENT
        # =========================================

        if selected_document != "All Documents":

            filtered_results = [
                r for r in semantic_results
                if r.chunk.doc_name.strip().lower()
                == selected_document.strip().lower()
            ]

            if len(filtered_results) > 0:
                semantic_results = filtered_results

        # =========================================
        # BM25
        # =========================================

        bm25_scores = bm25.get_scores_normalised(
            question
        )

        # =========================================
        # HYBRID SCORING
        # =========================================

        hybrid_results = []

        for result in semantic_results:

            chunk = result.chunk

            semantic_score = float(result.score)

            bm25_score = float(
                bm25_scores[chunk.chunk_index]
            )

            hybrid_score = (
                0.6 * semantic_score +
                0.4 * bm25_score
            )

            hybrid_results.append({
                "chunk": chunk,
                "semantic_score": semantic_score,
                "bm25_score": bm25_score,
                "hybrid_score": hybrid_score
            })

        hybrid_results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        top_results = hybrid_results[:4]

        # =========================================
        # CONTEXT
        # =========================================

        context = "\n\n".join([
            item["chunk"].text
            for item in top_results
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

Use the provided PDF context to answer.

If answer is not found in context say:
"I could not find that information in the uploaded PDFs."

========================
Conversation History
========================

{history_text}

========================
PDF Context
========================

{context}

========================
Question
========================

{question}
"""

        # =========================================
        # CITATIONS
        # =========================================

        citations = []

        for rank, item in enumerate(
            top_results,
            start=1
        ):

            chunk = item["chunk"]

            citations.append({
                "source": chunk.doc_name,
                "page": chunk.page_num,
                "rank": rank,
                "hybrid_score": round(
                    item["hybrid_score"],
                    3
                ),
                "preview": chunk.preview
            })

        # =========================================
        # STREAM GENERATOR
        # =========================================

        def generate():

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
                "temperature": 0.3,
                "stream": True
            }

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                stream=True
            )

            full_answer = ""

            for line in response.iter_lines():

                if line:

                    decoded = line.decode("utf-8")

                    if decoded.startswith("data: "):

                        data_str = decoded[6:]

                        if data_str == "[DONE]":
                            break

                        try:

                            data_json = json.loads(
                                data_str
                            )

                            delta = data_json["choices"][0]["delta"]

                            content = delta.get(
                                "content",
                                ""
                            )

                            if content:

                                full_answer += content

                                yield f"data: {json.dumps({'token': content})}\n\n"

                        except:
                            pass

            yield f"data: {json.dumps({'done': True, 'citations': citations})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream"
        )

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