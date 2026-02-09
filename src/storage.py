"""
storage.py — Rotating local evidence storage (images + JSON metadata).

Evidence structure on disk
──────────────────────────
  evidence/
  ├── 20250601/                     ← one folder per day
  │   ├── 143022_AB12CDE_frame.jpg  ← vehicle crop (JPEG, quality 85)
  │   ├── 143022_AB12CDE_plate.jpg  ← plate crop   (JPEG, quality 90)
  │   └── 143022_AB12CDE_meta.json  ← metadata (plate, timestamp, confidence…)
  ├── 20250602/
  │   └── …
  └── …

Rotation policy
───────────────
  1. Folders older than max_age_days are deleted entirely.
  2. If total evidence size exceeds max_size_mb, the oldest files are
     deleted one-by-one until we're back under the limit.

This ensures the SD card / SSD doesn't fill up even if the Pi runs
for weeks unattended.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class EvidenceStorage:
    """Save vehicle/plate crops organised by date; auto-rotate by age & size.

    Args:
        config: Full app config dict — we read the "storage" section.
    """

    def __init__(self, config: dict):
        cfg = config["storage"]
        self.root = Path(cfg["evidence_dir"])
        self.max_mb = cfg["max_size_mb"]
        self.max_days = cfg["max_age_days"]

        # Create the evidence directory if it doesn't exist
        # parents=True creates intermediate dirs;  exist_ok=True is not an error
        # if the directory already exists.
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info("Evidence dir: %s", self.root)

    # ------------------------------------------------------------------ #
    #  Saving evidence
    # ------------------------------------------------------------------ #

    def save(
        self,
        plate: str,
        frame: np.ndarray,
        plate_img: Optional[np.ndarray] = None,
        meta: Optional[dict] = None,
    ) -> str:
        """Persist a vehicle image, plate crop, and metadata to disk.

        Args:
            plate:     Normalised plate string (used in the filename).
            frame:     The vehicle crop (or full frame) as a BGR numpy array.
            plate_img: Optional cropped plate image (BGR numpy array).
            meta:      Optional extra metadata dict (merged into the JSON).

        Returns:
            Absolute path to the saved frame JPEG (used by the uploader).
        """
        ts = time.time()

        # Build date/time strings for directory and filename
        day = time.strftime("%Y%m%d", time.localtime(ts))    # e.g. "20250601"
        hms = time.strftime("%H%M%S", time.localtime(ts))    # e.g. "143022"

        # Create today's subdirectory
        day_dir = self.root / day
        day_dir.mkdir(exist_ok=True)

        # -- Save the vehicle / frame JPEG --
        fname = f"{hms}_{plate}_frame.jpg"
        fpath = day_dir / fname
        # cv2.imwrite params: [flag, value] — JPEG quality 85 (0-100)
        cv2.imwrite(str(fpath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        # -- Save the plate crop JPEG (if available) --
        pname = None
        if plate_img is not None and plate_img.size > 0:
            pname = f"{hms}_{plate}_plate.jpg"
            cv2.imwrite(
                str(day_dir / pname), plate_img, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )

        # -- Save metadata JSON --
        info = {
            "plate": plate,
            "timestamp": ts,
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            "frame_image": fname,
            "plate_image": pname,
            **(meta or {}),   # merge in caller's extra metadata
        }
        with open(day_dir / f"{hms}_{plate}_meta.json", "w") as f:
            json.dump(info, f, indent=2)

        return str(fpath)

    # ------------------------------------------------------------------ #
    #  Cleanup / rotation
    # ------------------------------------------------------------------ #

    def cleanup(self):
        """Run both age-based and size-based cleanup."""
        self._by_age()
        self._by_size()

    def _by_age(self):
        """Delete evidence directories older than max_age_days."""
        cutoff = time.time() - self.max_days * 86400  # 86400 seconds in a day

        for d in sorted(self.root.iterdir()):
            if d.is_dir():
                try:
                    if d.stat().st_mtime < cutoff:
                        shutil.rmtree(d)  # delete the entire directory tree
                        logger.info("Removed old evidence dir %s", d.name)
                except OSError as e:
                    logger.warning("Cleanup error %s: %s", d, e)

    def _by_size(self):
        """Delete oldest files until total evidence size is under max_size_mb."""
        limit = self.max_mb * 1024 * 1024  # convert MB to bytes

        # Walk the tree and collect all files with their modification time and size
        files = []
        total = 0
        for root, _, names in os.walk(self.root):
            for n in names:
                p = os.path.join(root, n)
                try:
                    s = os.path.getsize(p)
                    files.append((p, os.path.getmtime(p), s))
                    total += s
                except OSError:
                    pass

        if total <= limit:
            return  # within budget — nothing to do

        # Sort by modification time (oldest first) and delete until under limit
        files.sort(key=lambda x: x[1])
        for fp, _, sz in files:
            if total <= limit:
                break
            try:
                os.remove(fp)
                total -= sz
            except OSError:
                pass

        # Tidy up any empty directories left behind
        for d in self.root.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
