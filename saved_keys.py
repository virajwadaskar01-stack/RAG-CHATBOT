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

DEFAULT_KEYS = {
    "groq_api_key": "",
    "unsplash_api_key": "",
    "gemini_api_key": "",
}


def load_keys() -> dict:
    if not os.path.exists(KEYS_FILE):
        return dict(DEFAULT_KEYS)
    try:
        with open(KEYS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Fill in any missing keys (e.g. gemini_api_key added later) with defaults
        merged = dict(DEFAULT_KEYS)
        merged.update(data)
        return merged
    except (json.JSONDecodeError, IOError):
        return dict(DEFAULT_KEYS)


def save_keys(groq_api_key: str, unsplash_api_key: str, gemini_api_key: str = ""):
    with open(KEYS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "groq_api_key": groq_api_key,
                "unsplash_api_key": unsplash_api_key,
                "gemini_api_key": gemini_api_key,
            },
            f, ensure_ascii=False, indent=2,
        )
