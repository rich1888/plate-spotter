"""
main.py — Pipeline orchestrator for plate-spotter.

This is the main entry point.  It ties all the modules together:

  Camera  →  frame queue  →  Processing loop  →  SQLite queue  →  Upload thread
                               ├─ YOLO vehicle detection (every N-th frame)
                               ├─ IoU tracker (every frame)
                               ├─ Morphological plate localisation
                               ├─ Tesseract OCR
                               ├─ UK plate validation
                               └─ Dedup + evidence save

Threading model
───────────────
  Thread 1 (camera):      Grabs frames → puts them on frame_queue
  Thread 2 (main/this):   Reads frame_queue → detect → track → OCR → enqueue
  Thread 3 (uploader):    Reads SQLite queue → POST to API with retry

The processing work (detection + OCR) runs in the main thread because
Python's GIL (Global Interpreter Lock) means CPU-bound threads don't
truly run in parallel anyway.  The camera and upload threads are I/O-bound
so they benefit from threading.

Usage
─────
  # From the project directory:
  python -m src.main -c config.yaml

  # Or via systemd (see plate-spotter.service)
"""

import argparse
import logging
import signal
import time
from queue import Queue

from .config import load_config
from .camera import CameraCapture
from .detect import VehicleDetector, PlateLocalizer
from .track import VehicleTracker
from .ocr import PlateOCR
from .uk_plate import classify_plate
from .uploader import Uploader
from .queue_manager import UploadQueue
from .storage import EvidenceStorage

logger = logging.getLogger("plate-spotter")


class PlateSpotter:
    """Top-level application object — initialises components and runs the loop.

    Args:
        config_path: Path to config.yaml (or None for auto-discovery).
    """

    def __init__(self, config_path=None):
        # Load and merge configuration
        self.cfg = load_config(config_path)

        # Set up logging before anything else so all modules can log
        self._setup_logging()

        # ── Initialise all components ──────────────────────────────────

        # Camera capture thread
        self.camera = CameraCapture(self.cfg)

        # YOLO vehicle detector
        self.detector = VehicleDetector(self.cfg)

        # Morphological plate finder
        self.plate_loc = PlateLocalizer(self.cfg)

        # IoU tracker for deduplicating vehicles across frames
        self.tracker = VehicleTracker(
            iou_threshold=self.cfg["tracking"]["iou_threshold"],
            max_disappeared=self.cfg["tracking"]["max_disappeared"],
            max_tracks=self.cfg["tracking"]["max_tracks"],
        )

        # Tesseract OCR engine
        self.ocr = PlateOCR(self.cfg)

        # Local evidence storage (rotating image folders)
        self.storage = EvidenceStorage(self.cfg)

        # SQLite upload queue (persists across restarts)
        self.upload_queue = UploadQueue(
            self.cfg["storage"].get("queue_db", "./upload_queue.db")
        )

        # API uploader with dedup cache
        self.uploader = Uploader(self.cfg, self.upload_queue)

        # ── Internal state ────────────────────────────────────────────

        # Bounded queue between camera thread and processing loop
        self.frame_q: Queue = Queue(
            maxsize=self.cfg["pipeline"]["frame_queue_size"]
        )

        # How often to run YOLO (every N-th frame)
        self._det_interval = self.cfg["detection"]["detection_interval"]

        # Minimum frames a vehicle must be tracked before OCR fires
        self._min_stable = self.cfg["tracking"]["min_stable_frames"]

        # Flag to control the main loop
        self._running = False

        # ── Counters for periodic stats logging ───────────────────────
        self._frames = 0
        self._detections = 0
        self._ocr_ok = 0
        self._uploads = 0

    # ------------------------------------------------------------------ #
    #  Logging setup
    # ------------------------------------------------------------------ #

    def _setup_logging(self):
        """Configure Python logging with console + file handlers."""
        lvl_name = self.cfg["storage"].get("log_level", "INFO").upper()
        # getattr(logging, "INFO") returns the integer 20
        lvl = getattr(logging, lvl_name, logging.INFO)

        fmt = logging.Formatter(
            "%(asctime)s  %(name)-18s  %(levelname)-7s  %(message)s"
        )

        root = logging.getLogger()
        root.setLevel(lvl)

        # Console handler — always enabled
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

        # File handler — if a log_file path is configured
        log_file = self.cfg["storage"].get("log_file")
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            root.addHandler(fh)

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def run(self):
        """Start all workers and enter the main processing loop.

        Blocks until SIGINT (Ctrl-C) or SIGTERM is received.
        """
        self._running = True

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._sig)    # Ctrl-C
        signal.signal(signal.SIGTERM, self._sig)   # systemctl stop / kill

        logger.info("=== plate-spotter starting ===")
        logger.info("Device: %s", self.cfg["upload"]["device_id"])
        logger.info("API: %s", self.cfg["upload"]["api_url"])

        # Load the YOLO model (downloads on first run, then cached)
        self.detector.load()

        # Start the camera thread (begins pushing frames onto frame_q)
        self.camera.start(self.frame_q)

        # Start the upload thread (begins draining the SQLite queue)
        self.uploader.start()

        # Timestamp for periodic cleanup
        last_cleanup = time.time()

        try:
            # ── Main processing loop ──────────────────────────────────
            while self._running:
                # Read next frame from the camera queue (blocks up to 1 s)
                item = None
                try:
                    item = self.frame_q.get(timeout=1.0)
                except Exception:
                    continue  # timeout — loop around and check _running
                if item is None:
                    continue

                frame, ts, seq = item
                self._frames += 1

                # ── Vehicle detection (every N-th frame) ──────────────
                if seq % self._det_interval == 0:
                    # Run YOLO on this frame
                    dets = self.detector.detect(frame)
                    self._detections += len(dets)
                    # Feed detections to the tracker
                    self.tracker.update(dets)
                else:
                    # No detection this frame — just re-feed existing
                    # visible tracks so the tracker ages missing ones
                    self.tracker.update([
                        (t.bbox, t.confidence)
                        for t in self.tracker.tracks.values()
                        if t.frames_missing == 0
                    ])

                # ── OCR candidates (stable, un-read tracks) ───────────
                for trk in self.tracker.get_ocr_candidates(self._min_stable):
                    self._try_ocr(frame, ts, trk)

                # ── Periodic housekeeping (once per hour) ─────────────
                if time.time() - last_cleanup > 3600:
                    self.storage.cleanup()
                    last_cleanup = time.time()

                # ── Stats logging (every 300 frames) ──────────────────
                if self._frames % 300 == 0:
                    self._log_stats()

        except KeyboardInterrupt:
            pass  # already handled by signal handler
        finally:
            self.shutdown()

    # ------------------------------------------------------------------ #
    #  OCR + validation for a single tracked vehicle
    # ------------------------------------------------------------------ #

    def _try_ocr(self, frame, ts, trk):
        """Attempt plate localisation + OCR on a tracked vehicle.

        Steps:
          1. Crop the vehicle ROI from the frame.
          2. Run morphological plate localisation inside the ROI.
          3. Run Tesseract OCR on the plate crop.
          4. Validate the result against UK plate formats.
          5. If valid: save evidence, submit for upload (with dedup).

        Args:
            frame: Full camera frame (BGR numpy array).
            ts:    Unix timestamp of this frame.
            trk:   Track object for the vehicle.
        """
        # Mark OCR as attempted so we don't retry this track
        trk.ocr_attempted = True

        # Crop the vehicle region from the frame
        x1, y1, x2, y2 = trk.bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        # Try to find a plate-shaped region in the vehicle crop
        result = self.plate_loc.find(roi)
        if result is None:
            return
        plate_img, (px, py, pw, ph) = result

        # Run OCR on the plate crop
        ocr_out = self.ocr.read(plate_img)
        if ocr_out is None:
            return
        text, conf = ocr_out

        # Validate against UK plate formats
        info = classify_plate(
            text,
            min_length=self.cfg["uk_plate"]["min_length"],
            max_length=self.cfg["uk_plate"]["max_length"],
            reject_all_same=self.cfg["uk_plate"]["reject_all_same"],
        )
        if info is None:
            logger.debug("Rejected plate text: %s (conf=%.1f)", text, conf)
            return

        norm = info["normalized"]
        dateless = info["dateless"]

        logger.info(
            "PLATE  %-8s  fmt=%-20s  dateless=%-5s  conf=%.0f  raw=%s",
            norm, info["format"], dateless, conf, info["raw"],
        )
        self._ocr_ok += 1

        # Save evidence images + metadata to disk
        img_path = self.storage.save(
            norm, roi, plate_img,
            meta={
                "confidence": conf,
                "format": info["format"],
                "raw": info["raw"],
            },
        )

        # Submit for upload (the uploader checks dedup internally)
        if not self.uploader.is_duplicate(norm):
            self.uploader.submit(
                norm, dateless, ts, img_path,
                meta={"confidence": conf, "format": info["format"]},
            )
            self._uploads += 1

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _log_stats(self):
        """Print a summary of processing counters."""
        qs = self.upload_queue.stats()
        logger.info(
            "STATS  frames=%d  dets=%d  ocr_ok=%d  uploads=%d  queue=%s",
            self._frames, self._detections, self._ocr_ok, self._uploads, qs,
        )

    def _sig(self, signum, _frame):
        """Handle SIGINT / SIGTERM — signal the main loop to stop."""
        logger.info("Signal %d received — shutting down", signum)
        self._running = False

    def shutdown(self):
        """Cleanly stop all threads and log final stats."""
        logger.info("Shutting down …")
        self.camera.stop()
        self.uploader.stop()
        self._log_stats()
        logger.info("=== plate-spotter stopped ===")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Parse command-line arguments and run the application.

    Usage:
        python -m src.main -c config.yaml
    """
    ap = argparse.ArgumentParser(description="Raspberry Pi plate spotter")
    ap.add_argument(
        "-c", "--config",
        default=None,
        help="Path to config.yaml (default: auto-discover)",
    )
    args = ap.parse_args()
    PlateSpotter(args.config).run()


# This guard means main() only runs when you execute this file directly
# (python -m src.main), NOT when you import it from another module.
if __name__ == "__main__":
    main()
