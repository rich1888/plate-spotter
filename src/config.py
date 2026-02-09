"""
config.py — Configuration management for plate-spotter.

How it works:
  1. A big dictionary of sensible DEFAULT values lives in this file.
  2. When the app starts it loads your config.yaml and merges your
     values on top of the defaults (so you only need to override what
     you care about).
  3. Two environment variables can override secrets so they never
     need to live in a file:
       PLATE_SPOTTER_API_TOKEN
       PLATE_SPOTTER_DEVICE_ID

Typical usage:
    from src.config import load_config
    cfg = load_config("config.yaml")
    print(cfg["upload"]["api_url"])
"""

import os
from typing import Optional

# PyYAML — a library that reads/writes YAML files (human-friendly config format)
import yaml

# ---------------------------------------------------------------------------
# DEFAULT_CONFIG
# Every setting the app uses, with a safe default.  Your config.yaml only
# needs to contain the keys you want to change.
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # ── Camera ────────────────────────────────────────────────────────
    "camera": {
        "resolution": [1280, 720],   # Width x height in pixels
        "fps": 15,                   # Max frames per second to capture
        "format": "RGB888",          # Pixel format for picamera2
        "use_picamera2": True,       # False = use OpenCV webcam instead
        "opencv_device": 0,          # /dev/video0 — only used when picamera2 is off
    },

    # ── Vehicle detection (YOLOv8-nano) ───────────────────────────────
    "detection": {
        "model_path": "yolov8n.onnx",  # Pre-exported ONNX model (see README)
        "confidence": 0.4,           # Only keep detections above this score (0-1)
        "vehicle_classes": [2, 3, 5, 7],  # COCO class IDs:
                                          #   2 = car, 3 = motorcycle,
                                          #   5 = bus, 7 = truck
        "input_size": 640,           # Resize frames to this before YOLO (pixels).
                                     # Smaller = faster but less accurate.
        "detection_interval": 3,     # Run YOLO every N-th frame (skip the rest).
                                     # The tracker fills in the gaps.
    },

    # ── Plate localisation (OpenCV morphological — no model needed) ───
    "plate_detection": {
        "min_aspect_ratio": 2.0,     # Plate width / height — reject below this
        "max_aspect_ratio": 7.0,     # …and above this
        "min_area_ratio": 0.005,     # Plate area ÷ vehicle-ROI area — minimum
        "max_area_ratio": 0.15,      # …maximum
        "min_plate_height_px": 20,   # Ignore plate crops shorter than this
    },

    # ── Tracker ───────────────────────────────────────────────────────
    "tracking": {
        "max_disappeared": 30,       # Delete a track after this many missed frames
        "min_stable_frames": 5,      # Only OCR a vehicle seen for ≥ N frames
        "iou_threshold": 0.3,        # IoU overlap needed to match detection → track
        "max_tracks": 50,            # Cap to avoid runaway memory use
    },

    # ── OCR (Tesseract) ──────────────────────────────────────────────
    "ocr": {
        "confidence_threshold": 50,  # 0-100; reject OCR results below this
        "max_blur_variance": 100.0,  # Laplacian variance.  Below = too blurry.
        "tesseract_config": "--psm 7 --oem 3",
            # --psm 7 = treat image as a single text line
            # --oem 3 = use both legacy + LSTM engines
        "char_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            # Only allow these characters in output (avoids stray punctuation)
    },

    # ── UK plate validation ──────────────────────────────────────────
    "uk_plate": {
        "min_length": 2,             # Shortest valid plate string
        "max_length": 8,             # Longest valid plate string
        "reject_all_same": True,     # Reject "AAAA" or "1111" as garbage
    },

    # ── API upload ───────────────────────────────────────────────────
    "upload": {
        "api_url": "https://reghistory.com/api/pi/spotted",
        "api_token": "",             # Set via env var PLATE_SPOTTER_API_TOKEN
        "device_id": "pi1",          # Identifies this Pi in the API
        "timeout": 30,               # HTTP timeout in seconds
        "max_retries": 5,            # Give up after this many failures
        "retry_delay_seconds": 60,   # Base delay between retries (doubles each time)
    },

    # ── Deduplication ────────────────────────────────────────────────
    "dedupe": {
        "window_minutes": 5,         # Ignore the same plate for this many minutes
    },

    # ── Local storage ────────────────────────────────────────────────
    "storage": {
        "evidence_dir": "./evidence",      # Where to save vehicle/plate images
        "max_size_mb": 500,                # Auto-delete oldest when exceeded
        "max_age_days": 7,                 # Auto-delete evidence older than this
        "queue_db": "./upload_queue.db",   # SQLite file for the upload queue
        "log_file": "./plate-spotter.log", # Application log file
        "log_level": "INFO",               # DEBUG / INFO / WARNING / ERROR
    },

    # ── Pipeline tuning ──────────────────────────────────────────────
    "pipeline": {
        "frame_queue_size": 10,  # Max frames waiting for processing.
                                 # If full the camera drops frames — that's OK.
    },
}


def load_config(path: Optional[str] = None) -> dict:
    """Load configuration from a YAML file and merge with defaults.

    Lookup order when *path* is None:
      1. $PLATE_SPOTTER_CONFIG environment variable
      2. ./config.yaml  (next to the running script)
      3. /etc/plate-spotter/config.yaml  (system-wide)

    Args:
        path: Explicit path to a YAML file, or None for auto-discovery.

    Returns:
        A fully-populated config dictionary (defaults + your overrides).
    """
    # Start with a deep copy of defaults so we never mutate the originals
    config = _deep_copy(DEFAULT_CONFIG)

    # Auto-discover config file if no explicit path given
    if path is None:
        candidates = [
            os.environ.get("PLATE_SPOTTER_CONFIG", ""),
            "./config.yaml",
            "/etc/plate-spotter/config.yaml",
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                path = c
                break

    # Read the YAML file and merge on top of defaults
    if path and os.path.isfile(path):
        with open(path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        _deep_merge(config, user_config)

    # Environment-variable overrides — keeps secrets out of config files
    env_token = os.environ.get("PLATE_SPOTTER_API_TOKEN")
    if env_token:
        config["upload"]["api_token"] = env_token

    env_device = os.environ.get("PLATE_SPOTTER_DEVICE_ID")
    if env_device:
        config["upload"]["device_id"] = env_device

    return config


# ---------------------------------------------------------------------------
# Helper functions (private — the leading underscore is a Python convention
# meaning "don't import or call this from outside this file")
# ---------------------------------------------------------------------------

def _deep_copy(d: dict) -> dict:
    """Recursively copy a nested dictionary so mutations don't leak."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy(v)
        elif isinstance(v, list):
            out[k] = v.copy()  # shallow list copy is fine here (values are scalars)
        else:
            out[k] = v
    return out


def _deep_merge(base: dict, override: dict):
    """Recursively merge *override* into *base* in place.

    Sub-dictionaries are merged; everything else is replaced.
    """
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
