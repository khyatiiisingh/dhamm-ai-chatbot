import os
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# Ensure NLTK uses the correct path
nltk.data.path.append(r"C:\Users\khyati singh\AppData\Roaming\nltk_data")

# Load cleaned transcript
TRANSCRIPT_FILE = "cleaned_transcript.txt"
if not os.path.exists(TRANSCRIPT_FILE):
    raise FileNotFoundError(f"Transcript file '{TRANSCRIPT_FILE}' not found!")

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    cleaned_text = f.read().strip()

# Tokenize transcript into sentences
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

# Initialize FastAPI
app = FastAPI(title="LMS Chatbot API", description="Chatbot powered by FAISS and Gemini API.")

# Configure Gemini API
api_key = os.environ.get("GEMINI_API_KEY")  # Read API key from environment
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in environment variables!")
genai.configure(api_key=api_key)

# Request Model for FastAPI
class QueryRequest(BaseModel):
    query: str

# Function to retrieve relevant text using FAISS
def retrieve_relevant_text(query: str) -> str:
    query_embedding = embedding_model.encode(query).reshape(1, -1)
    _, result_indices = index.search(query_embedding, k=3)
    retrieved_texts = [chunk_map[i] for i in result_indices[0]]
    return "\n".join(retrieved_texts)

# Generate AI response
def generate_response(query: str) -> str:
    relevant_text = retrieve_relevant_text(query)

    prompt = f"""
    You are an AI tutor. Answer the following question based on the given lecture transcript:

    Lecture Context: {relevant_text}

    Question: {query}
    """

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)

    return response.text if response.text else "I'm sorry, I couldn't generate a response."

# Define FastAPI endpoint
@app.post("/chat")
async def chat(request: QueryRequest):
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    response = generate_response(query)
    return {"response": response}

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render assigns PORT dynamically
    uvicorn.run(app, host="0.0.0.0", port=port)
