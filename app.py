import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from pypdf import PdfReader
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# Page config
st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide")
st.title("🧠 DocMind")
st.caption("Upload a PDF and chat with it using AI")

# Initialize clients
@st.cache_resource
def load_clients():
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("docmind")
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
    return groq_client, index, embedder

groq_client, index, embedder = load_clients()

# Chunk helper
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

# Sidebar - PDF upload
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Reading and uploading to Pinecone..."):
                # Extract text
                reader = PdfReader(uploaded_file)
                pdf_text = "".join(page.extract_text() for page in reader.pages)
                
                # Chunk and embed
                chunks = chunk_text(pdf_text)
                vectors = []
                for i, chunk in enumerate(chunks):
                    embedding = embedder.encode(chunk).tolist()
                    vectors.append({
                        "id": f"chunk-{i}",
                        "values": embedding,
                        "metadata": {"text": chunk}
                    })
                
                # Upload to Pinecone
                index.upsert(vectors=vectors)
                st.session_state.pdf_loaded = True
                st.session_state.pdf_name = uploaded_file.name
                st.success(f"✅ {len(chunks)} chunks uploaded!")

    if st.session_state.get("pdf_loaded"):
        st.info(f"📄 Active: {st.session_state.pdf_name}")

# Chat area
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    if not st.session_state.get("pdf_loaded"):
        st.warning("Please upload and process a PDF first!")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Search Pinecone
                question_embedding = embedder.encode(prompt).tolist()
                results = index.query(vector=question_embedding, top_k=3, include_metadata=True)
                context = "\n\n---\n\n".join(m["metadata"]["text"] for m in results["matches"])

                # Ask Groq
                response = groq_client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=[
                        {"role": "system", "content": f"Answer using ONLY these document excerpts:\n\n{context}"},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})