"""Load/save ML models to disk + optional Google Cloud Storage."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("models")
_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(name: str) -> Path:
    return _MODELS_DIR / f"{name}.joblib"


def _meta_path(name: str) -> Path:
    return _MODELS_DIR / f"{name}.meta.json"


def save_model(name: str, model, metadata: Optional[dict] = None):
    """Save a model locally and optionally to GCS."""
    path = _model_path(name)
    joblib.dump(model, path)

    meta = metadata or {}
    meta["saved_at"] = datetime.now(timezone.utc).isoformat()
    meta["model_name"] = name
    _meta_path(name).write_text(json.dumps(meta, indent=2))

    logger.info(f"Saved model {name} to {path}")

    # Optional: upload to GCS if configured
    try:
        _upload_to_gcs(name)
    except Exception as e:
        logger.warning(f"GCS upload skipped for {name}: {e}")


def load_model(name: str):
    """Load a model from disk. Returns None if missing."""
    path = _model_path(name)
    if not path.exists():
        # Try downloading from GCS
        try:
            _download_from_gcs(name)
        except Exception as e:
            logger.info(f"Model {name} not available locally or in GCS: {e}")
            return None

    if not path.exists():
        return None

    return joblib.load(path)


def load_metadata(name: str) -> dict:
    """Load model metadata."""
    path = _meta_path(name)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def list_models() -> list[dict]:
    """List all available models with metadata."""
    models = []
    for f in _MODELS_DIR.glob("*.joblib"):
        name = f.stem
        models.append({
            "name": name,
            "path": str(f),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "metadata": load_metadata(name),
        })
    return models


def _upload_to_gcs(name: str):
    """Upload model and metadata to GCS bucket."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(settings.ML_GCS_BUCKET)

    for ext in ["joblib", "meta.json"]:
        local = _MODELS_DIR / f"{name}.{ext}"
        if local.exists():
            blob = bucket.blob(f"models/{name}.{ext}")
            blob.upload_from_filename(str(local))


def _download_from_gcs(name: str):
    """Download model and metadata from GCS bucket."""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(settings.ML_GCS_BUCKET)

    for ext in ["joblib", "meta.json"]:
        blob = bucket.blob(f"models/{name}.{ext}")
        local = _MODELS_DIR / f"{name}.{ext}"
        if blob.exists():
            blob.download_to_filename(str(local))
