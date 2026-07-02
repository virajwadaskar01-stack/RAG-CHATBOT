# 🤖 Hybrid RAG Chatbot

A chatbot that thinks before it answers — retrieving grounded, source-cited information from your documents when relevant, and falling back to general knowledge when it's not.

Unlike typical "chat with your PDF" projects that only answer from documents, this system **automatically decides** whether a question needs document context or general knowledge — using a confidence-based relevance check on every query.

---

## 🎯 Why this project is different

Most fresher RAG projects are either:
- A plain LLM wrapper (no document grounding), or
- A rigid RAG bot that *only* answers from documents (fails on anything outside scope)

This project solves that with a **hybrid retrieval pipeline**: every query is scored for relevance against the uploaded document's vector index. High-confidence matches trigger a grounded, source-cited RAG response. Low-confidence matches fall back to general conversational AI — so the bot never says "I don't know" for questions it can actually answer.

---

## ⚙️ How it works

```
User Question
     │
     ▼
Embed question → Search FAISS vector index
     │
     ▼
Confidence score ≥ threshold?
     │
     ├── YES → Retrieve top-k chunks → Ground LLM response → Answer + Source citation
     │
     └── NO  → Fall back to general LLM chat
```

**Pipeline breakdown:**
1. **Document ingestion** — PDFs are parsed and split into overlapping chunks (500 words, 100-word overlap) to preserve context across boundaries
2. **Embedding** — each chunk is converted to a vector using `sentence-transformers` (all-MiniLM-L6-v2)
3. **Vector storage & retrieval** — FAISS (`IndexFlatIP`) stores embeddings and performs cosine-similarity search
4. **Relevance gating** — a tuned similarity threshold (0.35) decides whether retrieved chunks are trustworthy enough to use
5. **Generation** — Groq's Llama 3.1 (8B) generates the final answer, with a stricter system prompt and lower temperature (0.3) in RAG mode to reduce hallucination

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM Inference | Groq API (Llama 3.1 8B) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| PDF Parsing | pypdf |
| Language | Python |

---

## ✨ Features

- 🔀 **Automatic hybrid mode switching** — RAG vs. general chat, decided per-query
- 📎 **Source citation** — every RAG answer names the exact document and page it came from
- 🎯 **Tunable relevance threshold** — confidence score visible in the UI for transparency
- 💬 **Conversational memory** — maintains context across turns in general chat mode
- 📄 **Multi-document support** — index multiple PDFs into a single searchable knowledge base

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/virajwadaskar01-stack/RAG-CHATBOT.git
cd RAG-CHATBOT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
Sign up at [console.groq.com](https://console.groq.com) and generate an API key.

### 4. Run the app
```bash
streamlit run app.py
```

Paste your Groq API key in the sidebar, upload a PDF (optional), and start chatting.

---

## 📸 Demo

**General knowledge query (no document needed):**
> Q: *What is the capital of France?*
> A: *The capital of France is Paris.*
> `Mode: General Chat`

**Document-grounded query:**
> Q: *What is the Cauchy-Euler equation?*
> A: *A linear differential equation of the form... [Source: Cauchy-Euler - Cauchy-Legendre Equation.pdf, Page 6]*
> `Mode: RAG (confidence: 0.60)`

**Same document loaded, unrelated query — correctly falls back:**
> Q: *Tell me a fun fact about space.*
> A: *The Andromeda Galaxy is approaching us at about 250,000 mph...*
> `Mode: General Chat`

---

## 🔮 Future Improvements

- [ ] Hybrid search (keyword + semantic) for better retrieval on exact-match queries
- [ ] Support for more file types (DOCX, TXT, web URLs)
- [ ] Persistent vector store (currently in-memory per session)
- [ ] Streaming responses for faster perceived latency

---

## 👤 Author

**Viraj Wadaskar**
AI/ML Engineering Fresher | [GitHub](https://github.com/virajwadaskar01-stack)

---

*Built as part of a structured AI/ML placement preparation roadmap.*
