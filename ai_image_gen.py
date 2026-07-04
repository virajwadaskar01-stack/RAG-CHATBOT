"""
ai_image_gen.py
------------------
Generates AI images from a text prompt using Pollinations.ai's free public API.
No API key or signup required - good fit for a free portfolio project.
"""

import requests
import urllib.parse


def generate_image(prompt: str, width: int = 768, height: int = 768) -> bytes:
    """
    Returns raw image bytes (PNG/JPEG) for the given prompt, or raises an
    exception if generation fails (caller should catch and show a message).
    """
    encoded_prompt = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content
