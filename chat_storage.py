"""
chat_storage.py
-----------------
Local file-based persistence for named chat sessions, scoped PER USER.

Structure of chat_sessions.json:
{
    "viraj": {
        "Physics Notes Q&A": {"chat_history": [...], "llm_history": [...]},
        "Untitled Chat - 03 Jul": {...}
    },
    "another_user": {
        ...
    }
}
"""

import json
import os

SESSIONS_FILE = "chat_sessions.json"


def _load_all():
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_all(data):
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_user_sessions(username: str):
    """Returns {session_name: {chat_history, llm_history}} for this user only."""
    return _load_all().get(username, {})


def save_session(username: str, session_name: str, chat_history, llm_history):
    data = _load_all()
    if username not in data:
        data[username] = {}
    data[username][session_name] = {
        "chat_history": chat_history,
        "llm_history": llm_history,
    }
    _save_all(data)


def delete_session(username: str, session_name: str):
    data = _load_all()
    if username in data and session_name in data[username]:
        del data[username][session_name]
        _save_all(data)


def get_session_names(username: str):
    return list(load_user_sessions(username).keys())
