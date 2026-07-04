"""
photo_editor.py
------------------
Basic photo editing operations using Pillow (PIL).
Takes image bytes in, returns edited image bytes out - keeps everything
in memory, no temp files needed.
"""

import io
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


def _load(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()


def resize(image_bytes: bytes, width: int, height: int) -> bytes:
    img = _load(image_bytes)
    img = img.resize((width, height))
    return _to_bytes(img)


def crop(image_bytes: bytes, left: int, top: int, right: int, bottom: int) -> bytes:
    img = _load(image_bytes)
    img = img.crop((left, top, right, bottom))
    return _to_bytes(img)


def rotate(image_bytes: bytes, degrees: float) -> bytes:
    img = _load(image_bytes)
    img = img.rotate(-degrees, expand=True)  # negative so positive degrees = clockwise
    return _to_bytes(img)


def grayscale(image_bytes: bytes) -> bytes:
    img = _load(image_bytes)
    img = ImageOps.grayscale(img)
    return _to_bytes(img)


def sepia(image_bytes: bytes) -> bytes:
    img = _load(image_bytes).convert("RGB")
    width, height = img.size
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            tr = int(0.393 * r + 0.769 * g + 0.189 * b)
            tg = int(0.349 * r + 0.686 * g + 0.168 * b)
            tb = int(0.272 * r + 0.534 * g + 0.131 * b)
            pixels[x, y] = (min(tr, 255), min(tg, 255), min(tb, 255))
    return _to_bytes(img)


def blur(image_bytes: bytes, intensity: float = 2.0) -> bytes:
    img = _load(image_bytes)
    img = img.filter(ImageFilter.GaussianBlur(radius=intensity))
    return _to_bytes(img)


def sharpen(image_bytes: bytes) -> bytes:
    img = _load(image_bytes)
    img = img.filter(ImageFilter.SHARPEN)
    return _to_bytes(img)


def adjust_brightness(image_bytes: bytes, factor: float) -> bytes:
    """factor: 1.0 = unchanged, <1 darker, >1 brighter"""
    img = _load(image_bytes)
    img = ImageEnhance.Brightness(img).enhance(factor)
    return _to_bytes(img)


def adjust_contrast(image_bytes: bytes, factor: float) -> bytes:
    img = _load(image_bytes)
    img = ImageEnhance.Contrast(img).enhance(factor)
    return _to_bytes(img)
