"""Binary/large-artifact storage: floor plan images, OCR output, GeoJSON, reports.

Backed by Supabase Storage buckets in production; falls back to a local
./dev_data/storage/<bucket>/ directory tree when Supabase isn't configured,
so exports work the same way in dev mode.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_SUPABASE_URL = os.environ.get("SUPABASE_URL")
_SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

_DEV_STORAGE_ROOT = Path(__file__).resolve().parent.parent.parent / "dev_data" / "storage"

BUCKETS = ["floorplans", "images", "ocr", "geojson", "reports", "screenshots", "youtube_frames"]


class ObjectStorage:
    def __init__(self) -> None:
        self.dev_mode = not (_SUPABASE_URL and _SUPABASE_KEY)
        if self.dev_mode:
            for bucket in BUCKETS:
                (_DEV_STORAGE_ROOT / bucket).mkdir(parents=True, exist_ok=True)
            self._client = None
        else:
            from supabase import create_client  # type: ignore

            self._client = create_client(_SUPABASE_URL, _SUPABASE_KEY)

    def put_bytes(self, bucket: str, path: str, data: bytes) -> str:
        if self.dev_mode:
            dest = _DEV_STORAGE_ROOT / bucket / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return str(dest)
        self._client.storage.from_(bucket).upload(path, data, {"upsert": "true"})
        return f"{bucket}/{path}"

    def put_json(self, bucket: str, path: str, obj) -> str:
        return self.put_bytes(bucket, path, json.dumps(obj, indent=2, default=str).encode("utf-8"))

    def get_bytes(self, bucket: str, path: str) -> bytes | None:
        if self.dev_mode:
            src = _DEV_STORAGE_ROOT / bucket / path
            return src.read_bytes() if src.exists() else None
        try:
            return self._client.storage.from_(bucket).download(path)
        except Exception:
            return None

    def get_json(self, bucket: str, path: str):
        data = self.get_bytes(bucket, path)
        return json.loads(data) if data is not None else None


_storage: ObjectStorage | None = None


def get_storage() -> ObjectStorage:
    global _storage
    if _storage is None:
        _storage = ObjectStorage()
    return _storage
