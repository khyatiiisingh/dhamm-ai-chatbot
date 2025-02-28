import faiss
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import nltk
from flask import Flask, request, jsonify

nltk.download("punkt", download_dir="/usr/local/nltk_data")
nltk.data.path.append("/usr/local/nltk_data")

# Load cleaned transcript
TRANSCRIPT_FILE = "cleaned_transcript.txt"
with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    cleaned_text = f.read()

from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(cleaned_text)

# Chunking the transcript
chunk_size = 500
chunks = []
current_chunk = ""

for sentence in sentences:
    if len(current_chunk) + len(sentence) < chunk_size:
        current_chunk += " " + sentence
    else:
        chunks.append(current_chunk.strip())
        current_chunk = sentence

if current_chunk:
    chunks.append(current_chunk.strip())

print(f"Total Chunks Created: {len(chunks)}")

# Load sentence transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings
query_embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks])

# Map chunks
chunk_map = {i: chunks[i] for i in range(len(chunks))}

# Store in FAISS
dimension = query_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(query_embeddings)

print(f"Stored {len(chunks)} chunks in FAISS!")

# Initialize Flask
app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyB4VqnPfl_Y98HL42QWLMXfQstpTZljzGY")

# Function to retrieve relevant text using FAISS
def retrieve_relevant_text(query):
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    _, result_indices = index.search(query_embedding, k=3)
    retrieved_texts = [chunk_map[i] for i in result_indices[0]]
    return "\n".join(retrieved_texts)

# Generate AI response
def generate_response(query):
    relevant_text = retrieve_relevant_text(query)

    prompt = f"""
    You are an AI tutor. Answer the following question based on the given lecture transcript:

    Lecture Context: {relevant_text}

    Question: {query}
    """

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    return response.text

# Define Flask API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    response = generate_response(query)
    return jsonify({"response": response})

# Correct Port Binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=True)
