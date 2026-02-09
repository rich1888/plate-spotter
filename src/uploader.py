"""
uploader.py — API upload with deduplication and background retry.

This module has two parts:

  DedupeCache
  ───────────
  A time-windowed in-memory cache.  If we've already uploaded plate
  "AB12CDE" in the last 5 minutes, we skip it.  This avoids hammering
  the API when the same car is stuck in traffic in front of the camera.

  Uploader
  ────────
  A background thread that:
    1. Polls the SQLite queue for pending items.
    2. Sends each one to the API as a multipart/form-data POST.
    3. Marks it as completed on success, or schedules a retry on failure.

API request format
──────────────────
  POST https://reghistory.com/api/pi/spotted
  Headers:
    Authorization: Bearer <token>
    X-Device-ID: <device_id>
  Form fields:
    plate      – normalised plate string
    dateless   – "true" or "false"
    timestamp  – Unix timestamp
    device_id  – Pi identifier
  File:
    image      – JPEG vehicle/plate crop
"""

import json
import logging
import os
import threading
import time
from typing import Dict, Optional

import requests

from .queue_manager import UploadQueue

logger = logging.getLogger(__name__)


class DedupeCache:
    """Time-windowed deduplication for normalised plate strings.

    Thread-safe: can be called from the processing thread (to check)
    and the upload thread (to record) simultaneously.

    Args:
        window_minutes: How long to remember a plate before allowing
                        it to be uploaded again.
    """

    def __init__(self, window_minutes: int = 5):
        self._window = window_minutes * 60  # convert to seconds
        self._seen: Dict[str, float] = {}   # plate → Unix timestamp
        self._lock = threading.Lock()

    def seen_recently(self, plate: str) -> bool:
        """Check if *plate* was seen within the dedup window.

        Also records the plate if it hasn't been seen.

        Returns:
            True if this is a duplicate (skip it).
            False if this is new (proceed with upload).
        """
        now = time.time()
        with self._lock:
            # Prune old entries to keep memory bounded
            self._seen = {
                p: t for p, t in self._seen.items()
                if now - t < self._window
            }
            if plate in self._seen:
                return True   # duplicate
            self._seen[plate] = now
            return False      # new plate


class Uploader:
    """Background thread that drains the SQLite queue to the RegHistory API.

    Args:
        config: Full app config dict.
        queue:  UploadQueue instance (shared with the processing thread).
    """

    def __init__(self, config: dict, queue: UploadQueue):
        cfg = config["upload"]
        self.api_url: str = cfg["api_url"]
        self.api_token: str = cfg["api_token"]
        self.device_id: str = cfg["device_id"]
        self.timeout: int = cfg["timeout"]
        self.max_retries: int = cfg["max_retries"]
        self.retry_delay: float = cfg["retry_delay_seconds"]

        self.queue = queue
        self.dedupe = DedupeCache(config["dedupe"]["window_minutes"])

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    #  Public interface
    # ------------------------------------------------------------------ #

    def is_duplicate(self, plate: str) -> bool:
        """Quick check — has this plate been uploaded recently?"""
        return self.dedupe.seen_recently(plate)

    def submit(
        self,
        plate: str,
        dateless: bool,
        ts: float,
        image_path: str,
        meta: Optional[dict] = None,
    ):
        """Check dedup and enqueue a plate for upload.

        If the plate was uploaded within the dedup window, this is a no-op.
        Otherwise the plate is added to the SQLite queue for the upload
        thread to process.

        Args:
            plate:      Normalised plate string.
            dateless:   True if the plate is dateless.
            ts:         Unix timestamp of the sighting.
            image_path: Path to the saved evidence image.
            meta:       Optional extra metadata dict.
        """
        if self.dedupe.seen_recently(plate):
            logger.debug("Dedup skip: %s", plate)
            return
        self.queue.enqueue(plate, dateless, ts, image_path, meta)

    # ------------------------------------------------------------------ #
    #  Thread lifecycle
    # ------------------------------------------------------------------ #

    def start(self):
        """Launch the background upload thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="uploader"
        )
        self._thread.start()
        logger.info("Uploader thread started")

    def stop(self):
        """Signal the upload thread to stop and wait for it."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Uploader thread stopped")

    # ------------------------------------------------------------------ #
    #  Upload loop (runs in background thread)
    # ------------------------------------------------------------------ #

    def _loop(self):
        """Poll the queue and upload items until stopped."""
        while self._running:
            try:
                items = self.queue.pending(limit=5)

                if not items:
                    time.sleep(2)  # nothing to do — sleep before polling again
                    continue

                for item in items:
                    if not self._running:
                        break
                    self._send(item)

                # Periodically clean old completed rows
                self.queue.cleanup()

            except Exception:
                logger.exception("Upload loop error")
                time.sleep(5)

    def _send(self, item: dict):
        """Send a single queue item to the API.

        On success: marks the item as 'completed'.
        On failure: calls queue.fail() which schedules a retry with
                    exponential backoff.

        Args:
            item: A dict representing one row from the queue table.
        """
        iid = item["id"]
        self.queue.set_status(iid, "uploading")

        fh = None  # file handle — we need to close it in the finally block
        try:
            # -- Build the HTTP request --------------------------------- #
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "X-Device-ID": self.device_id,
            }

            # Form fields
            data = {
                "plate": item["plate"],
                "dateless": str(bool(item["dateless"])).lower(),
                "timestamp": str(item["ts"]),
                "device_id": self.device_id,
            }

            # Merge in any extra metadata from the JSON column
            meta = json.loads(item.get("meta") or "{}")
            for k, v in meta.items():
                if k not in data:
                    data[k] = str(v)

            # Attach the image file if it exists
            files = {}
            img = item.get("image_path", "")
            if img and os.path.isfile(img):
                fh = open(img, "rb")
                files["image"] = (os.path.basename(img), fh, "image/jpeg")

            # -- Send the request --------------------------------------- #
            resp = requests.post(
                self.api_url,
                headers=headers,
                data=data,
                files=files,
                timeout=self.timeout,
            )
            resp.raise_for_status()  # raises an exception for 4xx/5xx

            # -- Success ------------------------------------------------ #
            self.queue.set_status(iid, "completed")
            logger.info(
                "Uploaded id=%d  plate=%s  http=%d",
                iid, item["plate"], resp.status_code,
            )

        except Exception:
            logger.exception("Upload failed id=%d plate=%s", iid, item["plate"])
            self.queue.fail(iid, self.max_retries, self.retry_delay)

        finally:
            # Always close the file handle to avoid leaking file descriptors
            if fh:
                fh.close()
