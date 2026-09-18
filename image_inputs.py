"""Decode and validate images attached to chat messages. Pillow only, no MLX.

Images arrive as base64 data URIs and are decoded here into PIL images.
Client strings are never handed to mlx_vlm: its load_image() also accepts
http(s) URLs and file paths, which a chat request must not be able to reach.
"""
from __future__ import annotations

import base64
import binascii
import re
from io import BytesIO
from typing import Any

MAX_IMAGES_PER_MESSAGE = 4
MAX_IMAGES_PER_REQUEST = 8
MAX_IMAGE_BYTES = 8 * 1024 * 1024
# Refuse to decode anything larger than this many pixels (decompression bombs).
MAX_SOURCE_PIXELS = 50_000_000
# Each 32x32 pixel block becomes one prompt token, so large images are slow to
# prefill. The browser downscales first; this is the server-side ceiling.
MAX_EDGE = 2048
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "GIF"}

_DATA_URI = re.compile(r"^data:image/(?:png|jpeg|webp|gif);base64,([A-Za-z0-9+/=\s]+)$")


def decode_image(data_uri: Any, *, max_edge: int = MAX_EDGE):
    """Return an RGB PIL image, or raise ValueError with a user-facing reason."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    if not isinstance(data_uri, str):
        raise ValueError("Each image must be a base64 data URI string")
    match = _DATA_URI.match(data_uri)
    if not match:
        raise ValueError("Images must be PNG, JPEG, WebP or GIF base64 data URIs")
    try:
        raw = base64.b64decode(match.group(1), validate=False)
    except (binascii.Error, ValueError):
        raise ValueError("Image data is not valid base64") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")

    try:
        image = Image.open(BytesIO(raw))
        if image.format not in ALLOWED_FORMATS:
            raise ValueError("Images must be PNG, JPEG, WebP or GIF")
        width, height = image.size
        if width < 1 or height < 1 or width * height > MAX_SOURCE_PIXELS:
            raise ValueError("Image dimensions are too large")
        image.seek(0)  # First frame of an animated GIF/WebP.
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA", "P"}:
            # Flatten transparency onto white instead of the black that convert() gives.
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise ValueError("Image could not be decoded") from None

    if max(image.size) > max_edge:
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image


def decode_message_images(raw_images: Any, already: int = 0) -> list:
    """Decode one message's 'images' array, enforcing per-message and per-request limits."""
    if raw_images is None:
        return []
    if not isinstance(raw_images, list):
        raise ValueError("'images' must be an array of data URIs")
    if len(raw_images) > MAX_IMAGES_PER_MESSAGE:
        raise ValueError(f"At most {MAX_IMAGES_PER_MESSAGE} images per message")
    if already + len(raw_images) > MAX_IMAGES_PER_REQUEST:
        raise ValueError(
            f"At most {MAX_IMAGES_PER_REQUEST} images per conversation can be sent to the model; "
            "start a new chat or remove older images"
        )
    return [decode_image(item) for item in raw_images]
