from dotenv import load_dotenv
import os
from groq import Groq
from pypdf import PdfReader

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- Step 1: Read and CHUNK the PDF ---
print("Reading PDF...")
reader = PdfReader("BMW - Wikipedia.pdf")
pdf_text = ""
for page in reader.pages:
    pdf_text += page.extract_text()

print(f"PDF loaded! {len(reader.pages)} pages, {len(pdf_text)} characters")

# Split into chunks of ~1000 characters with 100-char overlap
def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # overlap so we don't cut mid-sentence
    return chunks

chunks = chunk_text(pdf_text)
print(f"Split into {len(chunks)} chunks")
print("-" * 40)

# --- Step 2: Find relevant chunks (simple keyword search for now) ---
def find_relevant_chunks(question, chunks, top_n=6):
    question_words = question.lower().split()
    scored = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(1 for word in question_words if word in chunk_lower)
        scored.append((score, i, chunk))
    scored.sort(reverse=True)
    return [chunk for score, i, chunk in scored[:top_n] if score > 0]

# --- Step 3: Chat loop ---
print("Ask questions about the BMW PDF! Type 'quit' to exit.")
print()

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Goodbye!")
        break

    # Find most relevant chunks
    relevant = find_relevant_chunks(user_input, chunks)
    
    if not relevant:
        print("DocMind: I couldn't find anything relevant in the document for that question.\n")
        continue

    context = "\n\n---\n\n".join(relevant)

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. Answer the question using ONLY the document excerpts below. If the answer isn't in the excerpts, say so.\n\nDocument excerpts:\n{context}"},
            {"role": "user", "content": user_input}
        ]
    )
    
    ai_response = response.choices[0].message.content
    print(f"DocMind: {ai_response}\n")