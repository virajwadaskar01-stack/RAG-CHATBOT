"""
saved_keys.py
---------------
Stores API keys locally in a JSON file so they don't need to be re-typed
every time the app restarts.

Note: this is fine for a personal, local project like this one - the file
never leaves your computer and is excluded from GitHub via .gitignore.
It is NOT meant for a publicly hosted/multi-user deployment.
"""

import json
import os

KEYS_FILE = "saved_keys.json"


def load_keys() -> dict:
    if not os.path.exists(KEYS_FILE):
        return {"groq_api_key": "", "unsplash_api_key": ""}
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"groq_api_key": "", "unsplash_api_key": ""}


def save_keys(groq_api_key: str, unsplash_api_key: str):
    with open(KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"groq_api_key": groq_api_key, "unsplash_api_key": unsplash_api_key},
            f, ensure_ascii=False, indent=2,
        )
