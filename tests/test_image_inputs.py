from __future__ import annotations

import base64
import sys
import unittest
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from PIL import Image
except ImportError:  # The other test modules stay dependency-free.
    Image = None

import image_inputs
from image_inputs import decode_image, decode_message_images


def data_uri(image, fmt: str = "PNG", mime: str | None = None) -> str:
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    mime = mime or f"image/{fmt.lower()}"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode()


@unittest.skipIf(Image is None, "Pillow is not installed")
class ImageInputTests(unittest.TestCase):
    def test_png_and_jpeg_decode_to_rgb(self):
        for fmt in ("PNG", "JPEG"):
            image = decode_image(data_uri(Image.new("RGB", (40, 30), (200, 10, 10)), fmt))
            self.assertEqual((image.mode, image.size), ("RGB", (40, 30)))

    def test_transparency_is_flattened_onto_white(self):
        image = decode_image(data_uri(Image.new("RGBA", (8, 8), (0, 0, 0, 0))))
        self.assertEqual(image.getpixel((4, 4)), (255, 255, 255))

    def test_large_image_is_downscaled_keeping_aspect_ratio(self):
        image = decode_image(data_uri(Image.new("RGB", (4000, 1000)), "JPEG"))
        self.assertEqual(image.size, (image_inputs.MAX_EDGE, image_inputs.MAX_EDGE // 4))

    def test_urls_paths_and_other_types_are_rejected(self):
        for value in ("https://example.com/cat.png", "/etc/passwd", "file:///etc/passwd",
                      "data:image/svg+xml;base64,PHN2Zy8+", "data:text/html;base64,AAAA", 7, None):
            with self.assertRaises(ValueError):
                decode_image(value)

    def test_corrupt_and_mislabelled_data_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_image("data:image/png;base64," + base64.b64encode(b"not an image").decode())
        bmp = data_uri(Image.new("RGB", (4, 4)), "BMP", mime="image/png")
        with self.assertRaises(ValueError):
            decode_image(bmp)

    def test_oversized_payload_is_rejected_before_decoding(self):
        blob = base64.b64encode(b"\0" * (image_inputs.MAX_IMAGE_BYTES + 1)).decode()
        with self.assertRaisesRegex(ValueError, "larger than"):
            decode_image("data:image/png;base64," + blob)

    def test_message_and_request_limits(self):
        uri = data_uri(Image.new("RGB", (4, 4)))
        self.assertEqual(decode_message_images(None), [])
        self.assertEqual(len(decode_message_images([uri, uri])), 2)
        with self.assertRaisesRegex(ValueError, "per message"):
            decode_message_images([uri] * (image_inputs.MAX_IMAGES_PER_MESSAGE + 1))
        with self.assertRaisesRegex(ValueError, "per conversation"):
            decode_message_images([uri], already=image_inputs.MAX_IMAGES_PER_REQUEST)
        with self.assertRaises(ValueError):
            decode_message_images("not-a-list")


if __name__ == "__main__":
    unittest.main()
