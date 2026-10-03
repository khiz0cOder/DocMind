# 🧠 DocMind

A RAG-powered document assistant. Upload any PDF and chat with it, DocMind finds the relevant sections and answers your questions using an LLM, without hallucinating content that isn't there.

🔗 **Live demo:** [docminddd.streamlit.app](https://docminddd.streamlit.app)

---

## The problem it solves

LLMs are powerful but they don't know your documents. Dumping an entire PDF into a prompt doesn't scale. it hits token limits and loses accuracy. DocMind solves this with a proper retrieval pipeline: chunk, embed, store, retrieve, generate.

---

## How it works

1. PDF is extracted and split into overlapping chunks
2. Each chunk is embedded using Sentence Transformers
3. Vectors are stored in Pinecone
4. On each question, the query is embedded and the closest chunks are retrieved
5. Only the relevant chunks are passed to the LLM  keeping responses accurate and within token limits

---

## Stack

- **Groq** (Qwen 3.8B) — LLM inference
- **Pinecone** — vector storage and semantic search
- **Sentence Transformers** — text embeddings
- **Streamlit** — frontend
- **pypdf** — document parsing

---

## Run locally

```bash
git clone https://github.com/khiz0cOder/DocMind.git
cd DocMind
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Set environment variables:
GROQ_API_KEY=key
PINECONE_API_KEY=key
---

*Part of my AI Engineering portfolio. I'm a Cloud Engineer (2 years) moving into AI — building projects that use the same stack production teams use.*