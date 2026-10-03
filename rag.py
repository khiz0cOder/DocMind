from dotenv import load_dotenv
import os
from groq import Groq
from pypdf import PdfReader
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("docmind")

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Read and chunk PDF
print("Reading PDF...")
reader = PdfReader("BMW - Wikipedia.pdf")
pdf_text = ""
for page in reader.pages:
    pdf_text += page.extract_text()

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(pdf_text)
print(f"Split into {len(chunks)} chunks")

# Embed and upload chunks to Pinecone
print("Uploading to Pinecone... (first time only)")
vectors = []
for i, chunk in enumerate(chunks):
    embedding = embedder.encode(chunk).tolist()
    vectors.append({
        "id": f"chunk-{i}",
        "values": embedding,
        "metadata": {"text": chunk}
    })

index.upsert(vectors=vectors)
print(f"Uploaded {len(vectors)} vectors to Pinecone!")
print("-" * 40)
print("Ask questions about the BMW PDF! Type 'quit' to exit.")
print()

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    # Embed the question and search Pinecone
    question_embedding = embedder.encode(user_input).tolist()
    results = index.query(vector=question_embedding, top_k=3, include_metadata=True)
    
    # Get relevant chunks
    relevant_chunks = [match["metadata"]["text"] for match in results["matches"]]
    context = "\n\n---\n\n".join(relevant_chunks)

    # Ask Groq
    response = groq_client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": f"Answer using ONLY these document excerpts:\n\n{context}"},
            {"role": "user", "content": user_input}
        ]
    )

    print(f"DocMind: {response.choices[0].message.content}\n")