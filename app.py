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
from datetime import datetime
import streamlit as st

from rag_engine import RAGEngine
from llm_client import LLMClient
import chat_storage

# ---- Config ----
RELEVANCE_THRESHOLD = 0.35  # tuned by experimentation - tweak based on testing
TOP_K = 3

st.set_page_config(page_title="QueAssist", page_icon="💬")

# ---- Custom CSS for WhatsApp-style aligned chat bubbles ----
st.markdown("""
<style>
.chat-row {
    display: flex;
    margin: 10px 0;
}
.chat-row.user {
    justify-content: flex-end;
}
.chat-row.bot {
    justify-content: flex-start;
}
.bubble {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 16px;
    font-size: 15px;
    line-height: 1.5;
    background-color: #000000;
}
.bubble.user {
    color: #FF9999;
    border: 1px solid #FF6B6B;
    border-bottom-right-radius: 4px;
}
.bubble.bot {
    color: #A0D8FF;
    border: 1px solid #4F8BF9;
    border-bottom-left-radius: 4px;
}
.mode-caption {
    font-size: 12px;
    color: #999;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 QueAssist")
st.caption("✨ Ask me anything — I'll dig into your documents 📄 when relevant, "
           "and chat like a normal AI 🤖 otherwise.")

# ---- Session state setup ----
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if "current_session_name" not in st.session_state:
    st.session_state.current_session_name = None  # None = unsaved new chat

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm_history" not in st.session_state:
    st.session_state.llm_history = []

# ---- Sidebar: API key + file upload ----
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("🔑 Groq API Key", type="password",
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

    st.divider()
    st.header("💬 My Chats")

    # ---- New Chat button ----
    if st.button("➕ New Chat", use_container_width=True):
        # If there's an unsaved conversation with messages, auto-save it first
        # so nothing gets lost. It can be renamed later.
        if st.session_state.chat_history and not st.session_state.current_session_name:
            auto_name = f"Untitled Chat - {datetime.now().strftime('%d %b, %I:%M %p')}"
            chat_storage.save_session(
                auto_name,
                st.session_state.chat_history,
                st.session_state.llm_history,
            )
        st.session_state.chat_history = []
        st.session_state.llm_history = []
        st.session_state.current_session_name = None
        st.rerun()

    # ---- Save current chat with a custom name ----
    if st.session_state.chat_history:
        default_name = st.session_state.current_session_name or ""
        chat_name_input = st.text_input("💾 Save chat as:", value=default_name,
                                         placeholder="e.g. Physics Notes Q&A")
        if st.button("Save Chat", use_container_width=True):
            if chat_name_input.strip():
                chat_storage.save_session(
                    chat_name_input.strip(),
                    st.session_state.chat_history,
                    st.session_state.llm_history,
                )
                st.session_state.current_session_name = chat_name_input.strip()
                st.success(f"Saved as '{chat_name_input.strip()}' ✅")
            else:
                st.warning("Please enter a name first.")

    # ---- List of saved chats ----
    if "renaming_chat" not in st.session_state:
        st.session_state.renaming_chat = None

    saved_names = chat_storage.get_session_names()
    if saved_names:
        st.caption("📂 Saved chats")
        for name in saved_names:
            if st.session_state.renaming_chat == name:
                # Show rename input inline
                new_name = st.text_input("New name:", value=name, key=f"rename_input_{name}")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ Confirm", key=f"confirm_{name}", use_container_width=True):
                        if new_name.strip() and new_name.strip() != name:
                            sessions = chat_storage.load_all_sessions()
                            chat_storage.save_session(
                                new_name.strip(),
                                sessions[name]["chat_history"],
                                sessions[name]["llm_history"],
                            )
                            chat_storage.delete_session(name)
                            if st.session_state.current_session_name == name:
                                st.session_state.current_session_name = new_name.strip()
                        st.session_state.renaming_chat = None
                        st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel", key=f"cancel_{name}", use_container_width=True):
                        st.session_state.renaming_chat = None
                        st.rerun()
            else:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    if st.button(f"💬 {name}", key=f"open_{name}", use_container_width=True):
                        sessions = chat_storage.load_all_sessions()
                        st.session_state.chat_history = sessions[name]["chat_history"]
                        st.session_state.llm_history = sessions[name]["llm_history"]
                        st.session_state.current_session_name = name
                        st.rerun()
                with col2:
                    if st.button("✏️", key=f"rename_{name}"):
                        st.session_state.renaming_chat = name
                        st.rerun()
                with col3:
                    if st.button("🗑️", key=f"delete_{name}"):
                        chat_storage.delete_session(name)
                        if st.session_state.current_session_name == name:
                            st.session_state.chat_history = []
                            st.session_state.llm_history = []
                            st.session_state.current_session_name = None
                        st.rerun()
    else:
        st.caption("No saved chats yet — start chatting and save it! 📝")

# ---- Main chat display ----
def render_message(role, content, mode=None):
    css_class = "user" if role == "user" else "bot"
    icon = "🧑" if role == "user" else "🤖"
    mode_html = f'<div class="mode-caption">⚙️ Mode: {mode}</div>' if mode else ""
    html = (
        f'<div class="chat-row {css_class}">'
        f'<div class="bubble {css_class}">{icon} {content}{mode_html}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


for msg in st.session_state.chat_history:
    render_message(msg["role"], msg["content"], msg.get("mode"))

# ---- Chat input ----
user_query = st.chat_input("Ask a question... 💭")

if user_query:
    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar first.")
        st.stop()

    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    render_message("user", user_query)

    llm = LLMClient(api_key=api_key)

    # ---- HYBRID DECISION LOGIC ----
    mode = "General Chat 💬"
    answer = ""

    with st.spinner("Thinking... 🧠"):
        if st.session_state.rag_engine.has_documents():
            results = st.session_state.rag_engine.retrieve(user_query, top_k=TOP_K)
            top_score = results[0][2] if results else 0.0

            if top_score >= RELEVANCE_THRESHOLD:
                mode = f"RAG 📚 (confidence: {top_score:.2f})"
                answer = llm.rag_chat(user_query, results)
            else:
                answer = llm.general_chat(user_query, st.session_state.llm_history)
        else:
            answer = llm.general_chat(user_query, st.session_state.llm_history)

    # Show assistant message
    render_message("assistant", answer, mode)

    st.session_state.chat_history.append({"role": "assistant", "content": answer, "mode": mode})
    st.session_state.llm_history.append({"role": "user", "content": user_query})
    st.session_state.llm_history.append({"role": "assistant", "content": answer})

    # Auto-save only if this chat has already been named/saved once
    if st.session_state.current_session_name:
        chat_storage.save_session(
            st.session_state.current_session_name,
            st.session_state.chat_history,
            st.session_state.llm_history,
        )
