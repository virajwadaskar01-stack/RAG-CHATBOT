"""
app.py
-------
Streamlit UI for QueAssist - a hybrid RAG + general chat assistant with
per-user login and saved chat history.

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
import auth
import saved_keys
import doc_generator
import image_search
import ai_image_gen
import photo_editor
import video_editor

# ---- Config ----
RELEVANCE_THRESHOLD = 0.35  # tuned by experimentation - tweak based on testing
TOP_K = 3

st.set_page_config(page_title="QueAssist", page_icon="💬")

# ---- Custom CSS ----
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

# ---- Auth session state ----
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "guest_mode" not in st.session_state:
    st.session_state.guest_mode = False


# =====================================================================
# LOGIN / SIGNUP SCREEN (shown only if not logged in)
# =====================================================================
def show_login_screen():
    st.title("💬 QueAssist")
    st.caption("Log in to save and revisit your chats — or just continue as a guest to ask a quick question.")

    tab_login, tab_signup = st.tabs(["🔑 Log In", "🆕 Sign Up"])

    with tab_login:
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Log In", use_container_width=True):
            success, message = auth.verify_login(login_username, login_password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = login_username.strip().lower()
                st.rerun()
            else:
                st.error(message)

    with tab_signup:
        signup_username = st.text_input("Choose a username", key="signup_username")
        signup_password = st.text_input("Choose a password", type="password", key="signup_password")
        if st.button("Create Account", use_container_width=True):
            success, message = auth.register_user(signup_username, signup_password)
            if success:
                st.success(f"{message} You can now log in from the 'Log In' tab.")
            else:
                st.error(message)

    st.divider()
    st.caption("Just want to ask something quick? Your chat won't be saved, but no account needed.")
    if st.button("👤 Continue as Guest", use_container_width=True):
        st.session_state.guest_mode = True
        st.rerun()


if not st.session_state.logged_in and not st.session_state.guest_mode:
    show_login_screen()
    st.stop()  # Don't render the rest of the app until logged in or guest mode chosen


# =====================================================================
# MAIN APP (only reached once logged in)
# =====================================================================
st.title("💬 QueAssist")
if st.session_state.logged_in:
    st.caption(f"✨ Logged in as **{st.session_state.username}** — ask me anything, "
               "I'll use your documents 📄 when relevant.")
else:
    st.caption("✨ Browsing as a guest — ask me anything! "
               "Log in if you'd like to save this chat for later.")

# ---- Session state setup ----
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

if "current_session_name" not in st.session_state:
    st.session_state.current_session_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm_history" not in st.session_state:
    st.session_state.llm_history = []

username = st.session_state.username  # None if guest

# ---- Sidebar ----
with st.sidebar:
    if st.session_state.logged_in:
        st.write(f"👤 **{username}**")
        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.guest_mode = False
            st.session_state.chat_history = []
            st.session_state.llm_history = []
            st.session_state.current_session_name = None
            st.rerun()
    else:
        st.write("👤 **Guest**")
        st.caption("Chats won't be saved in guest mode.")
        if st.button("🔑 Log In / Sign Up", use_container_width=True):
            st.session_state.guest_mode = False
            st.session_state.chat_history = []
            st.session_state.llm_history = []
            st.session_state.current_session_name = None
            st.rerun()

    st.divider()
    st.header("⚙️ Setup")

    _saved = saved_keys.load_keys()
    api_key = st.text_input("🔑 Groq API Key", type="password",
                             value=_saved.get("groq_api_key", ""),
                             help="Get a free key at console.groq.com")

    st.divider()
    st.header("🖼️ Images")
    st.caption("Photos are added automatically when they'd help answer your question.")
    unsplash_key = st.text_input(
        "Unsplash API Key", type="password",
        value=_saved.get("unsplash_api_key", ""),
        help="Get a free key at unsplash.com/developers"
    )

    # Auto-save keys locally so they don't need to be re-entered next time
    if api_key != _saved.get("groq_api_key", "") or unsplash_key != _saved.get("unsplash_api_key", ""):
        saved_keys.save_keys(api_key, unsplash_key)

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
    st.header("📝 Generate Document")

    doc_source = st.radio(
        "Generate from:",
        ["A topic I specify", "This conversation"],
        key="doc_source",
    )

    doc_topic = ""
    if doc_source == "A topic I specify":
        doc_topic = st.text_input("Topic:", placeholder="e.g. Machine Learning basics")

    doc_format = st.selectbox("Format:", ["PDF", "PPT", "Word (DOCX)"])

    if st.button("✨ Generate", use_container_width=True):
        if not api_key:
            st.error("⚠️ Please enter your Groq API key first.")
        elif doc_source == "A topic I specify" and not doc_topic.strip():
            st.warning("Please enter a topic first.")
        elif doc_source == "This conversation" and not st.session_state.chat_history:
            st.warning("There's no conversation yet to generate from.")
        else:
            with st.spinner("Creating your document... 📄"):
                try:
                    llm = LLMClient(api_key=api_key)

                    if doc_source == "A topic I specify":
                        outline = llm.generate_outline("topic", doc_topic.strip())
                        base_filename = doc_topic.strip().replace(" ", "_")[:40]
                    else:
                        convo_text = "\n".join(
                            f"{m['role'].upper()}: {m['content']}"
                            for m in st.session_state.chat_history
                        )
                        outline = llm.generate_outline("conversation", convo_text)
                        base_filename = "QueAssist_Conversation"

                    if doc_format == "PDF":
                        file_bytes = doc_generator.generate_pdf(outline)
                        mime = "application/pdf"
                        ext = "pdf"
                    elif doc_format == "PPT":
                        file_bytes = doc_generator.generate_pptx(outline)
                        mime = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                        ext = "pptx"
                    else:
                        file_bytes = doc_generator.generate_docx(outline)
                        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        ext = "docx"

                    st.success("Document ready! ✅")
                    st.download_button(
                        label=f"⬇️ Download {doc_format}",
                        data=file_bytes,
                        file_name=f"{base_filename}.{ext}",
                        mime=mime,
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Something went wrong generating the document: {e}")

    st.divider()

    if st.session_state.logged_in:
        st.header("💬 My Chats")

        # ---- New Chat button ----
        if st.button("➕ New Chat", use_container_width=True):
            if st.session_state.chat_history and not st.session_state.current_session_name:
                auto_name = f"Untitled Chat - {datetime.now().strftime('%d %b, %I:%M %p')}"
                chat_storage.save_session(
                    username, auto_name,
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
                        username, chat_name_input.strip(),
                        st.session_state.chat_history,
                        st.session_state.llm_history,
                    )
                    st.session_state.current_session_name = chat_name_input.strip()
                    st.success(f"Saved as '{chat_name_input.strip()}' ✅")
                else:
                    st.warning("Please enter a name first.")

        # ---- List of saved chats (for this user only) ----
        if "renaming_chat" not in st.session_state:
            st.session_state.renaming_chat = None

        saved_names = chat_storage.get_session_names(username)
        if saved_names:
            st.caption("📂 Your saved chats")
            for name in saved_names:
                if st.session_state.renaming_chat == name:
                    new_name = st.text_input("New name:", value=name, key=f"rename_input_{name}")
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("✅ Confirm", key=f"confirm_{name}", use_container_width=True):
                            if new_name.strip() and new_name.strip() != name:
                                sessions = chat_storage.load_user_sessions(username)
                                chat_storage.save_session(
                                    username, new_name.strip(),
                                    sessions[name]["chat_history"],
                                    sessions[name]["llm_history"],
                                )
                                chat_storage.delete_session(username, name)
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
                            sessions = chat_storage.load_user_sessions(username)
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
                            chat_storage.delete_session(username, name)
                            if st.session_state.current_session_name == name:
                                st.session_state.chat_history = []
                                st.session_state.llm_history = []
                                st.session_state.current_session_name = None
                            st.rerun()
        else:
            st.caption("No saved chats yet — start chatting and save it! 📝")
    else:
        st.info("🔒 Log in to save and revisit your chats.")


# ---- Helper: decide if a question would benefit from images ----
VISUAL_KEYWORDS = [
    "photo", "picture", "image", "show me", "look like", "what does",
    "visualize", "see a", "see the", "appearance", "scenery", "landscape",
]
SKIP_KEYWORDS = [
    "code", "function", "error", "debug", "syntax", "calculate", "formula",
    "equation", "write a", "explain the concept",
]


def question_wants_images(query: str) -> bool:
    q = query.lower()
    if any(skip in q for skip in SKIP_KEYWORDS):
        return False
    if any(kw in q for kw in VISUAL_KEYWORDS):
        return True
    # Default heuristic: places, people, animals, objects, events often benefit
    # from a visual, so allow it unless it looks like a technical/code question.
    return True


# ---- Chat input (kept at top level so it stays pinned to the bottom of the page) ----
user_query = st.chat_input("Ask a question... 💭")

# ---- Main tabs ----
tab_chat, tab_ai_image, tab_photo, tab_video = st.tabs(
    ["💬 Chat", "🎨 AI Image Generator", "🖌️ Photo Editor", "🎬 Video Editor"]
)

with tab_chat:
    # ---- Chat display ----
    def render_message(role, content, mode=None, images=None):
        css_class = "user" if role == "user" else "bot"
        icon = "🧑" if role == "user" else "🤖"
        mode_html = f'<div class="mode-caption">⚙️ Mode: {mode}</div>' if mode else ""
        html = (
            f'<div class="chat-row {css_class}">'
            f'<div class="bubble {css_class}">{icon} {content}{mode_html}</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)

        if images:
            cols = st.columns(len(images))
            for col, img in zip(cols, images):
                with col:
                    st.image(img["url"], use_column_width=True)
                    st.caption(f"📸 {img['credit']}")


    for msg in st.session_state.chat_history:
        render_message(msg["role"], msg["content"], msg.get("mode"), msg.get("images"))

    if user_query:
        if not api_key:
            st.error("⚠️ Please enter your Groq API key in the sidebar first.")
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        render_message("user", user_query)

        llm = LLMClient(api_key=api_key)

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

        render_message("assistant", answer, mode)

        # ---- Fetch related images automatically when it makes sense ----
        images = []
        if unsplash_key and question_wants_images(user_query):
            images = image_search.search_images(user_query, unsplash_key, count=3)
            if images:
                cols = st.columns(len(images))
                for col, img in zip(cols, images):
                    with col:
                        st.image(img["url"], use_column_width=True)
                        st.caption(f"📸 {img['credit']}")

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer, "mode": mode, "images": images
        })
        st.session_state.llm_history.append({"role": "user", "content": user_query})
        st.session_state.llm_history.append({"role": "assistant", "content": answer})

        if st.session_state.current_session_name:
            chat_storage.save_session(
                username, st.session_state.current_session_name,
                st.session_state.chat_history,
                st.session_state.llm_history,
            )

with tab_ai_image:
    st.subheader("🎨 AI Image Generator")
    st.caption("Free AI image generation powered by Pollinations.ai — no API key needed.")

    image_prompt = st.text_area("Describe the image you want:",
                                 placeholder="e.g. a futuristic city at sunset, digital art style")
    col_w, col_h = st.columns(2)
    with col_w:
        img_width = st.selectbox("Width", [512, 768, 1024], index=1)
    with col_h:
        img_height = st.selectbox("Height", [512, 768, 1024], index=1)

    if st.button("✨ Generate Image", use_container_width=True):
        if not image_prompt.strip():
            st.warning("Please describe what you'd like to see.")
        else:
            with st.spinner("Generating your image... 🎨 (can take 10-30 seconds)"):
                try:
                    img_bytes = ai_image_gen.generate_image(image_prompt.strip(), img_width, img_height)
                    st.image(img_bytes, use_column_width=True)
                    st.download_button(
                        "⬇️ Download Image", data=img_bytes,
                        file_name="queassist_generated.png", mime="image/png",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Image generation failed: {e}")

with tab_photo:
    st.subheader("🖌️ Photo Editor")
    st.caption("Upload a photo and apply quick edits.")

    uploaded_photo = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="photo_upload")

    if uploaded_photo is not None:
        photo_bytes = uploaded_photo.read()
        st.image(photo_bytes, caption="Original", use_column_width=True)

        edit_option = st.selectbox(
            "Choose an edit:",
            ["Grayscale", "Sepia", "Blur", "Sharpen", "Brightness", "Contrast",
             "Rotate", "Resize", "Crop"],
        )

        result_bytes = None

        if edit_option == "Grayscale":
            if st.button("Apply", use_container_width=True):
                result_bytes = photo_editor.grayscale(photo_bytes)

        elif edit_option == "Sepia":
            if st.button("Apply", use_container_width=True):
                result_bytes = photo_editor.sepia(photo_bytes)

        elif edit_option == "Blur":
            intensity = st.slider("Blur intensity", 1.0, 10.0, 2.0)
            if st.button("Apply", use_container_width=True):
                result_bytes = photo_editor.blur(photo_bytes, intensity)

        elif edit_option == "Sharpen":
            if st.button("Apply", use_container_width=True):
                result_bytes = photo_editor.sharpen(photo_bytes)

        elif edit_option == "Brightness":
            factor = st.slider("Brightness factor", 0.1, 3.0, 1.2)
            if st.button("Apply", use_container_width=True):
                result_bytes = photo_editor.adjust_brightness(photo_bytes, factor)

        elif edit_option == "Contrast":
            factor = st.slider("Contrast factor", 0.1, 3.0, 1.2)
            if st.button("Apply", use_container_width=True):
                result_bytes = photo_editor.adjust_contrast(photo_bytes, factor)

        elif edit_option == "Rotate":
            degrees = st.slider("Rotate degrees (clockwise)", 0, 360, 90)
            if st.button("Apply", use_container_width=True):
                result_bytes = photo_editor.rotate(photo_bytes, degrees)

        elif edit_option == "Resize":
            col_w, col_h = st.columns(2)
            with col_w:
                new_w = st.number_input("Width", min_value=10, value=400)
            with col_h:
                new_h = st.number_input("Height", min_value=10, value=400)
            if st.button("Apply", use_container_width=True):
                result_bytes = photo_editor.resize(photo_bytes, int(new_w), int(new_h))

        elif edit_option == "Crop":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                left = st.number_input("Left", min_value=0, value=0)
            with col2:
                top = st.number_input("Top", min_value=0, value=0)
            with col3:
                right = st.number_input("Right", min_value=1, value=300)
            with col4:
                bottom = st.number_input("Bottom", min_value=1, value=300)
            if st.button("Apply", use_container_width=True):
                result_bytes = photo_editor.crop(photo_bytes, int(left), int(top), int(right), int(bottom))

        if result_bytes:
            st.image(result_bytes, caption="Edited", use_column_width=True)
            st.download_button(
                "⬇️ Download Edited Photo", data=result_bytes,
                file_name="edited_photo.png", mime="image/png",
                use_container_width=True,
            )

with tab_video:
    st.subheader("🎬 Video Editor")
    st.caption("Upload a video and apply a basic edit. Processing can take a while depending on video length.")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video_upload")

    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        st.video(video_bytes)

        video_edit_option = st.selectbox(
            "Choose an edit:", ["Trim", "Resize", "Add Caption", "Extract Audio"]
        )

        result_video_bytes = None
        result_mime = "video/mp4"
        result_ext = "mp4"

        if video_edit_option == "Trim":
            col_s, col_e = st.columns(2)
            with col_s:
                start_sec = st.number_input("Start (seconds)", min_value=0.0, value=0.0)
            with col_e:
                end_sec = st.number_input("End (seconds)", min_value=0.1, value=5.0)
            if st.button("Apply", use_container_width=True):
                with st.spinner("Trimming video... 🎬"):
                    try:
                        result_video_bytes = video_editor.trim_video(video_bytes, start_sec, end_sec)
                    except Exception as e:
                        st.error(f"Trim failed: {e}")

        elif video_edit_option == "Resize":
            col_w, col_h = st.columns(2)
            with col_w:
                vid_w = st.number_input("Width", min_value=64, value=640)
            with col_h:
                vid_h = st.number_input("Height", min_value=64, value=360)
            if st.button("Apply", use_container_width=True):
                with st.spinner("Resizing video... 🎬"):
                    try:
                        result_video_bytes = video_editor.resize_video(video_bytes, int(vid_w), int(vid_h))
                    except Exception as e:
                        st.error(f"Resize failed: {e}")

        elif video_edit_option == "Add Caption":
            caption_text = st.text_input("Caption text:", placeholder="e.g. My awesome video")
            if st.button("Apply", use_container_width=True):
                if not caption_text.strip():
                    st.warning("Please enter caption text.")
                else:
                    with st.spinner("Adding caption... 🎬"):
                        try:
                            result_video_bytes = video_editor.add_caption(video_bytes, caption_text.strip())
                        except Exception as e:
                            st.error(f"Adding caption failed: {e}")

        elif video_edit_option == "Extract Audio":
            if st.button("Apply", use_container_width=True):
                with st.spinner("Extracting audio... 🎧"):
                    try:
                        result_video_bytes = video_editor.extract_audio(video_bytes)
                        result_mime = "audio/mpeg"
                        result_ext = "mp3"
                    except Exception as e:
                        st.error(f"Audio extraction failed: {e}")

        if result_video_bytes:
            st.success("Done! ✅")
            if result_ext == "mp3":
                st.audio(result_video_bytes)
            else:
                st.video(result_video_bytes)
            st.download_button(
                f"⬇️ Download {result_ext.upper()}", data=result_video_bytes,
                file_name=f"edited_video.{result_ext}", mime=result_mime,
                use_container_width=True,
            )
