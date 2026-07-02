"""
app.py
-------
Streamlit UI for the Hybrid RAG Chatbot.

Hybrid logic:
- If the user has uploaded a document AND the retrieved chunks have a
  similarity score above RELEVANCE_THRESHOLD -> answer using RAG (grounded).
- Otherwise -> fall back to general chat (plain LLM knowledge).
"""

import os
import tempfile
import streamlit as st

from rag_engine import RAGEngine
from llm_client import LLMClient

# ---- Config ----
RELEVANCE_THRESHOLD = 0.35  # tuned by experimentation - tweak based on testing
TOP_K = 3

st.set_page_config(page_title="Hybrid RAG Chatbot", page_icon="🤖")
st.title("🤖 Hybrid RAG Chatbot")
st.caption("Ask me anything — I'll use your uploaded documents when relevant, "
           "and fall back to general knowledge otherwise.")

# ---- Session state setup ----
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # for display: list of {"role", "content"}

if "llm_history" not in st.session_state:
    st.session_state.llm_history = []  # for API: list of {"role", "content"}

# ---- Sidebar: API key + file upload ----
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Groq API Key", type="password",
                             help="Get a free key at console.groq.com")

    st.divider()
    st.header("📄 Upload Documents")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Reading and indexing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                num_chunks = st.session_state.rag_engine.build_index(
                    tmp_path, source_name=uploaded_file.name
                )
                os.unlink(tmp_path)
            st.success(f"Indexed {num_chunks} chunks from '{uploaded_file.name}'")

    if st.session_state.rag_engine.has_documents():
        st.info(f"📚 {len(st.session_state.rag_engine.chunks)} chunks loaded and ready")

# ---- Main chat display ----
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("mode"):
            st.caption(f"_Mode: {msg['mode']}_")

# ---- Chat input ----
user_query = st.chat_input("Ask a question...")

if user_query:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
        st.stop()

    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    llm = LLMClient(api_key=api_key)

    # ---- HYBRID DECISION LOGIC ----
    mode = "General Chat"
    answer = ""

    if st.session_state.rag_engine.has_documents():
        results = st.session_state.rag_engine.retrieve(user_query, top_k=TOP_K)
        top_score = results[0][2] if results else 0.0

        if top_score >= RELEVANCE_THRESHOLD:
            mode = f"RAG (confidence: {top_score:.2f})"
            answer = llm.rag_chat(user_query, results)
        else:
            answer = llm.general_chat(user_query, st.session_state.llm_history)
    else:
        answer = llm.general_chat(user_query, st.session_state.llm_history)

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(f"_Mode: {mode}_")

    st.session_state.chat_history.append({"role": "assistant", "content": answer, "mode": mode})
    st.session_state.llm_history.append({"role": "user", "content": user_query})
    st.session_state.llm_history.append({"role": "assistant", "content": answer})
