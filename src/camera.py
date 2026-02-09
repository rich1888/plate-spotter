"""
camera.py — Threaded camera capture using picamera2 (or OpenCV fallback).

How it works:
  1. A background thread continuously grabs frames from the camera.
  2. Each frame is put into a bounded Queue (a thread-safe FIFO buffer).
  3. The processing thread (in main.py) reads frames from that queue.
  4. If the queue is full (processing is too slow), new frames are
     silently dropped — this prevents RAM from growing without bound.

picamera2 vs OpenCV
───────────────────
  - On a Raspberry Pi with a Pi Camera Module, picamera2 talks directly
    to the libcamera stack and is the recommended way to capture.
  - On a laptop/desktop (for development/testing), we fall back to
    OpenCV's VideoCapture which works with any USB webcam.
  - Set camera.use_picamera2 = false in config.yaml to force the fallback.
"""

import logging
import threading
import time
from queue import Queue, Full
from typing import Optional

import cv2
import numpy as np

# Python's logging module — gives us timestamped, levelled log messages.
# Every module creates its own logger with __name__ so you can tell which
# file a message came from.
logger = logging.getLogger(__name__)


class CameraCapture:
    """Captures frames in a background thread and puts them on a queue.

    Args:
        config: The full application config dict (we read the "camera" section).
    """

    def __init__(self, config: dict):
        cfg = config["camera"]
        self.resolution = tuple(cfg["resolution"])   # e.g. (1280, 720)
        self.fps = cfg["fps"]                         # target frames per second
        self.use_picamera2 = cfg.get("use_picamera2", True)
        self.opencv_device = cfg.get("opencv_device", 0)
        self.fmt = cfg.get("format", "RGB888")

        # Internal state
        self._camera = None                           # set in _init_*()
        self._running = False                         # controls the capture loop
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    #  Start / Stop
    # ------------------------------------------------------------------ #

    def start(self, frame_queue: Queue):
        """Launch the capture thread.

        Args:
            frame_queue: A bounded Queue to put (frame, timestamp, seq) tuples into.
        """
        self._running = True
        # daemon=True means this thread dies automatically if the main program exits
        self._thread = threading.Thread(
            target=self._loop, args=(frame_queue,), daemon=True, name="camera"
        )
        self._thread.start()
        logger.info(
            "Camera started  res=%s  fps=%d  backend=%s",
            self.resolution,
            self.fps,
            "picamera2" if self.use_picamera2 else "opencv",
        )

    def stop(self):
        """Signal the capture thread to stop and wait for it to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)  # wait up to 5 s for clean exit
        self._close()
        logger.info("Camera stopped")

    # ------------------------------------------------------------------ #
    #  Camera initialisation (private)
    # ------------------------------------------------------------------ #

    def _init_picamera2(self):
        """Set up the Raspberry Pi Camera Module via picamera2."""
        from picamera2 import Picamera2

        cam = Picamera2()
        # create_preview_configuration sets up a capture mode with the given
        # resolution and pixel format.  buffer_count=4 keeps a small ring
        # buffer so we don't lose frames during brief processing spikes.
        cam_cfg = cam.create_preview_configuration(
            main={"size": self.resolution, "format": self.fmt},
            buffer_count=4,
        )
        cam.configure(cam_cfg)
        cam.start()
        time.sleep(1.0)  # give auto-exposure / auto-white-balance time to settle
        self._camera = ("picam", cam)
        logger.info("picamera2 initialised")

    def _init_opencv(self):
        """Set up a webcam via OpenCV (used on laptops or when picamera2 is off)."""
        cap = cv2.VideoCapture(self.opencv_device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.opencv_device}")
        self._camera = ("cv", cap)
        logger.info("OpenCV VideoCapture initialised  dev=%s", self.opencv_device)

    def _close(self):
        """Release the camera hardware."""
        if self._camera is None:
            return
        kind, obj = self._camera
        try:
            if kind == "picam":
                obj.stop()
                obj.close()
            else:
                obj.release()
        except Exception:
            pass
        self._camera = None

    # ------------------------------------------------------------------ #
    #  Capture loop (runs in background thread)
    # ------------------------------------------------------------------ #

    def _loop(self, q: Queue):
        """Continuously grab frames and push them onto the queue.

        Each item on the queue is a tuple:
            (frame, timestamp, sequence_number)

        - frame:    a numpy array of shape (height, width, 3) in BGR colour order
        - timestamp: Unix time (seconds since 1970) when the frame was captured
        - sequence_number: integer counter starting at 1
        """
        # Try picamera2 first; if it's not installed, fall back to OpenCV
        if self.use_picamera2:
            try:
                self._init_picamera2()
            except (ImportError, RuntimeError) as exc:
                logger.warning(
                    "picamera2 unavailable (%s), falling back to OpenCV", exc
                )
                self.use_picamera2 = False
                self._init_opencv()
        else:
            self._init_opencv()

        # How long to wait between frames to hit the target FPS
        interval = 1.0 / self.fps
        seq = 0

        while self._running:
            t0 = time.monotonic()   # high-resolution timer

            frame = self._grab()
            if frame is None:
                time.sleep(0.05)
                continue

            seq += 1

            # put_nowait: try to add to the queue without waiting.
            # If the queue is full, Full is raised and we silently drop the frame.
            try:
                q.put_nowait((frame, time.time(), seq))
            except Full:
                pass  # processing thread is behind — just drop this frame

            # Sleep the remaining time to maintain the target FPS
            elapsed = time.monotonic() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

        logger.info("Capture loop exited after %d frames", seq)

    def _grab(self) -> Optional[np.ndarray]:
        """Grab a single frame from the camera. Returns None on failure."""
        if self._camera is None:
            return None
        kind, obj = self._camera
        try:
            if kind == "picam":
                # picamera2 returns RGB; OpenCV uses BGR, so we convert
                rgb = obj.capture_array()
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                ok, frame = obj.read()
                return frame if ok else None
        except Exception as exc:
            logger.error("Grab error: %s", exc)
            return None
