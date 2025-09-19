"""
Flask api get questin then convert embedding to get similar text in db then answer.
"""

import os
import uuid
import logging

from flask import Flask, request, jsonify


try:
    from embeddings import generate_embedding, generate_completion
    from search import (
        search_scipy_cosine,
        search_faiss_l2,
        search_faiss_dot,
        search_canberra,
        compare_top_ids,
        fetch_chunk_by_id
    )
except Exception:
    from .embeddings import generate_embedding, generate_completion
    from .search import (
        search_scipy_cosine,
        search_faiss_l2,
        search_faiss_dot,
        search_canberra,
        compare_top_ids,
        fetch_chunk_by_id
    )


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- Flask App ----------------
app = Flask(__name__)


@app.route("/")
def home():
    # return "Welcome to the Media Assist API"
    return jsonify({
        "message": "Welcome to the [WHO Publications](https://www.who.int/europe/publications/i) Assist API",
        "status": "ok"
    }), 200


@app.route("/question", methods=["POST"])
def handle_question():
    try:
        conversation_id = str(uuid.uuid4())

        # data = request.json
        data = request.get_json(force=True, silent=True) or {}
        logger.info(f"Incoming question request: {data}")

        question = data.get("question")
        top_k = data.get("top_k", 5)
        if not question:
            return jsonify({"error": "Question must be a non-empty string"}), 400


        # Step 1: Embed the query into a vector
        query_vector = generate_embedding(question)

        # Step 2: Run multiple search methods
        pg = search_scipy_cosine(query_vector, top_n=top_k)
        #display_results(pg, "Cosine Similarity (Postgres)")
        faiss_l2 = search_faiss_l2(query_vector, top_n=top_k)
        #display_results(faiss_l2, "Euclidean Distance (FAISS)")
        faiss_dot = search_faiss_dot(query_vector, top_n=top_k)
        #display_results(faiss_dot, "Dot Product (FAISS)")
        canberra = search_canberra(query_vector, top_n=top_k)
        #display_results(canberra, "Canberra Distance (Canberra)")

        # Step 3: Combine rankings (ensemble)
        top_ids = compare_top_ids(pg, faiss_l2, faiss_dot, canberra, top_n=top_k)
        if not top_ids:
            return jsonify({"error": "No results found in Database."}), 500

        # Step 4: Fetch top context chunks from DB
        # context_chunks = []
        # for cid, group_id, score in top_ids:
        #     text = fetch_chunk_by_id(cid)
        #     if text:
        #         context_chunks.append(text)

        # context_text = "\n\n".join(context_chunks)
        best_id = top_ids[0][0]
        source_link = top_ids[0][-1]
        context = fetch_chunk_by_id(best_id)
        answers = ""
        if context:
            answers += f"\n[Best Match: ID {best_id} link {source_link}]"
        else:
           answers += "Best chunk not found in DB."

        # Step 5: Generate model response
        answers += "\n=== Assistant Answer ===\n"
        # Call RAG model
        answer = generate_completion(question, context)
        answers += answer

        result = {
            "conversation_id": conversation_id,
            "question": question,
            "answer": answers,
            # "model_used": answer_data.get("model_used", "unknown"),
        }
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Unexpected error in /question")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🔄 Flask starting...")
    app.run(host='0.0.0.0', port=5000, debug=True)