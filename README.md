# QueAssist

QueAssist is a chatbot I built that can either answer general questions like a normal AI assistant, or answer questions from a PDF you upload — and it automatically figures out which one to do based on how relevant your question is to the document.

I built this as part of my AI/ML placement preparation, mainly to actually understand how RAG (Retrieval-Augmented Generation) works instead of just reading about it. Most tutorials show you a chatbot that only answers from a document, but I wanted something that felt more like a real assistant — so I added logic that checks a confidence score for every question and decides whether to pull from the document or just answer normally.

## What it does

- You can chat with it like a normal AI (powered by Llama 3.1 through Groq's API)
- You can upload a PDF, and it'll answer questions from that PDF instead, with the exact page it got the answer from
- It automatically switches between the two modes depending on the question — you don't have to tell it which mode to use
- You can save conversations with your own names, come back to them later, rename them, or start a new chat without losing the old one
- Custom UI with the questions and answers shown in separate colored bubbles

## How the RAG part works

When you upload a PDF, I split the text into overlapping chunks (so an answer doesn't get cut off between chunks), turn each chunk into a vector using a sentence-transformer model, and store all of them in a FAISS index.

When you ask a question, it gets converted into a vector too, and I search the FAISS index for the closest matching chunks. Each match comes with a similarity score. If that score is high enough (I settled on 0.35 after testing a bunch of questions), the app treats it as relevant and builds an answer using only that context. If it's below that, it just answers normally without touching the document.

This was actually the trickiest part to get right — too low a threshold and it forces the document into answers that have nothing to do with it, too high and it ignores the document even when it clearly has the answer. I tested it with random general questions alongside document-specific ones to tune the number.

## Tech I used

- Python
- Streamlit for the UI
- Groq API (Llama 3.1 8B) for generating responses
- sentence-transformers for turning text into embeddings
- FAISS for the vector search
- pypdf for reading PDFs

## Running it yourself

```bash
git clone https://github.com/virajwadaskar01-stack/RAG-CHATBOT.git
cd RAG-CHATBOT
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Groq API key from console.groq.com — the app asks for it in the sidebar, it's not stored anywhere in the code.

## Things I'd add if I kept working on this

- Right now the vector index only exists while the app is running — if you restart it, you have to re-upload your PDF. I'd want to save that to disk too.
- Better search that combines keyword matching with the semantic search, since right now it can miss exact terms/names if the wording is too different.
- Support for other file types, not just PDF.

## A couple of things I ran into while building this

The Groq Python library had a version conflict with a library called httpx that took me a bit to figure out — turned out I needed to pin `httpx==0.27.2` specifically. Also spent a while debugging why HTML tags like `</div>` were showing up as visible text in the chat — turned out to be Streamlit reading indented multi-line HTML as a code block instead of rendering it, so I had to flatten it into single-line strings.

---

Built by Viraj Wadaskar
