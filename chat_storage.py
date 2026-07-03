"""
chat_storage.py
-----------------
Local file-based persistence for MULTIPLE named chat sessions.
All sessions are stored in a single JSON file, keyed by session name.

Structure of chat_sessions.json:
{
    "My First Chat": {"chat_history": [...], "llm_history": [...]},
    "Physics Notes Q&A": {"chat_history": [...], "llm_history": [...]}
}
"""

import json
import os

SESSIONS_FILE = "chat_sessions.json"


def load_all_sessions():
    """Returns a dict of {session_name: {chat_history, llm_history}}."""
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_session(session_name, chat_history, llm_history):
    """Saves/overwrites a single named session, keeping all other sessions intact."""
    sessions = load_all_sessions()
    sessions[session_name] = {
        "chat_history": chat_history,
        "llm_history": llm_history,
    }
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


def delete_session(session_name):
    sessions = load_all_sessions()
    if session_name in sessions:
        del sessions[session_name]
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)


def get_session_names():
    return list(load_all_sessions().keys())
