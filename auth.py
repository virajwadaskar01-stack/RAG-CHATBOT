"""
auth.py
--------
Simple local authentication using salted password hashing (PBKDF2-SHA256).
No plaintext passwords are ever stored — only a random salt + hash per user.

Users are stored in users.json:
{
    "viraj": {"salt": "...", "hash": "..."}
}
"""

import json
import os
import hashlib
import secrets

USERS_FILE = "users.json"


def _load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_password(password: str, salt: str) -> str:
    """PBKDF2-HMAC-SHA256 with 100,000 iterations - a strong, standard-library-only choice."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


def username_exists(username: str) -> bool:
    return username.strip().lower() in _load_users()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password can't be empty."
    if len(password) < 4:
        return False, "Password should be at least 4 characters."

    users = _load_users()
    if username in users:
        return False, "That username is already taken."

    salt = secrets.token_hex(16)
    hashed = _hash_password(password, salt)
    users[username] = {"salt": salt, "hash": hashed}
    _save_users(users)
    return True, "Account created!"


def verify_login(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    users = _load_users()

    if username not in users:
        return False, "No account found with that username."

    salt = users[username]["salt"]
    expected_hash = users[username]["hash"]
    actual_hash = _hash_password(password, salt)

    if secrets.compare_digest(actual_hash, expected_hash):
        return True, "Login successful!"
    return False, "Incorrect password."
