import os
import torch
from flask import Flask, request, jsonify
from embedding_service import useEmdeb

_threads = int(os.environ.get("TORCH_THREADS", "2"))
torch.set_num_threads(_threads)

app = Flask(__name__)

useEmdeb()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "intfloat/multilingual-e5-base"})


@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Invalid 'text' value"}), 400

    embedding = useEmdeb().generate_embedding(text)
    return jsonify({"embedding": embedding, "dimension": len(embedding)})


@app.route("/embed/batch", methods=["POST"])
def embed_batch():
    data = request.get_json(force=True, silent=True)
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400

    texts = data["texts"]
    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400

    embeddings = useEmdeb().generate_embeddings_batch(texts)
    return jsonify({"embeddings": embeddings, "count": len(embeddings)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
