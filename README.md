# QueAssist

QueAssist is an AI assistant I built that started out as a simple RAG chatbot and grew into a small suite of AI tools. At its core, it can answer general questions like a normal AI assistant, or answer questions from a PDF you upload — automatically switching between the two depending on how relevant your question is to the document. On top of that, it now has its own login system, saved chat history, AI image generation, photo/video editing, and document export.

I built this as part of my AI/ML placement preparation, mainly to actually understand how these pieces work under the hood instead of just reading about them.

## What it does

**Chat**
- Chat with it like a normal AI (powered by Llama 3.1 through Groq's API)
- Upload a PDF and it'll answer questions from that PDF instead, citing the exact page it got the answer from
- Automatically switches between general chat and document-grounded answers based on a confidence score — you never have to tell it which mode to use
- Related photos show up automatically under an answer when the question is genuinely visual (like "tell me about the Eiffel Tower"), and stay out of the way for technical or conversational questions

**Accounts and history**
- Log in with a username and password (passwords are hashed with PBKDF2-SHA256, salted, never stored in plain text), or just continue as a guest if you don't want to save anything
- Save chats under your own names, reopen them later, rename them, or start a new chat without losing the old one — it gets auto-saved as "Untitled Chat" if you forget to name it

**Media Studio**
- Generate AI images from a text prompt (free, via Pollinations.ai, no key required)
- Edit photos you upload: grayscale, sepia, blur, sharpen, brightness/contrast, rotate, resize, crop
- Edit videos you upload: trim, resize, add a text caption, or extract the audio

**Document generation**
- Turn a topic or the current conversation into a downloadable PDF, PowerPoint, or Word document — the AI writes a structured outline first, then it gets converted into the actual file

## How the RAG part works

When you upload a PDF, I split the text into overlapping chunks (so an answer doesn't get cut off between chunks), turn each chunk into a vector using a sentence-transformer model, and store all of them in a FAISS index.

When you ask a question, it gets converted into a vector too, and I search the FAISS index for the closest matching chunks. Each match comes with a similarity score. If that score is high enough (I settled on 0.35 after testing), the app treats it as relevant and builds an answer using only that context. If it's below that, it just answers normally without touching the document.

Getting that threshold right was the trickiest part — too low and it forces the document into answers that have nothing to do with it, too high and it ignores the document even when it clearly has the answer.

## Tech I used

- Python, Streamlit for the UI
- Groq API (Llama 3.1 8B) for chat and outline generation
- sentence-transformers for embeddings, FAISS for vector search
- pypdf for reading PDFs
- fpdf2, python-docx, python-pptx for document generation
- Pillow for photo editing, MoviePy for video editing
- Unsplash API for related photos, Pollinations.ai for AI image generation

## Running it yourself

```bash
git clone https://github.com/virajwadaskar01-stack/RAG-CHATBOT.git
cd RAG-CHATBOT
pip install -r requirements.txt
streamlit run app.py
```

You'll need a free Groq API key from console.groq.com, and optionally a free Unsplash API key from unsplash.com/developers if you want the automatic photo feature. Both get entered in the sidebar and are saved locally on your machine so you don't have to re-type them every time — they're never uploaded to GitHub.

## A security lesson I learned building this

Early on, I accidentally pushed some personal files to GitHub — my local chat history and a file with hashed passwords — before I'd set up `.gitignore` properly. Once I noticed, I had to go back and actually remove them from git history (not just delete them going forward), fix a `.gitignore` file that Windows had silently saved without its leading dot, and regenerate an API key that had briefly been visible in a screenshot. It was a good reminder that `.gitignore` only stops future commits — it doesn't undo ones that already happened.

## Things I'd add if I kept working on this

- Right now the vector index only exists while the app is running — if you restart it, you have to re-upload your PDF.
- Better search that combines keyword matching with the semantic search, since it can miss exact terms if the wording is too different.
- Video editing is functional but slow, since it processes the full file through MoviePy/ffmpeg — could look into faster or streaming-based approaches.
- Support for other file types beyond PDF.

---

Built by Viraj Wadaskar
