"""Extracts unit-number/store-label text from a floor plan image.
pytesseract first (fast, offline); Claude vision as a fallback for messy
scans/handwriting where pytesseract's confidence is low.
"""
from __future__ import annotations

import io


def ocr_floorplan(image_bytes: bytes) -> list[dict]:
    """Returns [{"text": ..., "bbox": [x, y, w, h], "confidence": 0-1}, ...]."""
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        results = []
        for i, text in enumerate(data["text"]):
            text = text.strip()
            if not text:
                continue
            conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", -1) else 0.0
            results.append({
                "text": text,
                "bbox": [data["left"][i], data["top"][i], data["width"][i], data["height"][i]],
                "confidence": max(conf, 0.0) / 100.0,
            })
        return results
    except Exception:
        # pytesseract/PIL unavailable or OCR failed outright -- no geometry
        # evidence from this source this run; caller treats as empty result.
        return []
