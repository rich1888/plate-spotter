"""
track.py — Lightweight IoU-based multi-object tracker for vehicles.

What is tracking and why do we need it?
───────────────────────────────────────
The YOLO detector runs on individual frames — it doesn't know that the
red car in frame 10 is the *same* red car in frame 11.  The tracker
solves this by:

  1. Assigning each new detection a unique ID.
  2. Matching detections across frames using bounding-box overlap (IoU).
  3. Counting how many consecutive frames an object has been visible.

This matters because:
  - We only OCR a vehicle once it's "stable" (visible for N frames).
  - We never OCR the same tracked vehicle twice.
  - Avoids wasting CPU re-reading the same plate every frame.

IoU (Intersection over Union)
─────────────────────────────
A simple metric for how much two rectangles overlap:
  IoU = area_of_overlap / area_of_union
  0.0 = no overlap,  1.0 = perfect overlap.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Type alias for bounding boxes: (x1, y1, x2, y2) — top-left and bottom-right corners
BBox = Tuple[int, int, int, int]


@dataclass
class Track:
    """Represents a single tracked vehicle across multiple frames.

    Attributes:
        track_id:        Unique integer ID for this track.
        bbox:            Current bounding box (x1, y1, x2, y2) in pixels.
        confidence:      YOLO detection confidence for the latest match.
        frames_seen:     How many frames this object has been detected in.
        frames_missing:  Consecutive frames where the object was NOT detected.
        first_seen:      Unix timestamp of first detection.
        last_seen:       Unix timestamp of most recent detection.
        ocr_attempted:   True once we've tried to read the plate (success or not).
        ocr_result:      The plate text if OCR succeeded, else None.
        best_frame:      The full frame image when this track was best quality.
        best_plate_img:  The cropped plate image (for evidence storage).
        best_plate_bbox: Bounding box of the plate within the vehicle ROI.
        best_confidence: Highest YOLO confidence seen for this track.
    """
    track_id: int
    bbox: BBox
    confidence: float
    frames_seen: int = 1
    frames_missing: int = 0
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    ocr_attempted: bool = False
    ocr_result: Optional[str] = None
    best_frame: Optional[np.ndarray] = None
    best_plate_img: Optional[np.ndarray] = None
    best_plate_bbox: Optional[BBox] = None
    best_confidence: float = 0.0


def _iou(a: BBox, b: BBox) -> float:
    """Compute Intersection-over-Union between two bounding boxes.

    Each box is (x1, y1, x2, y2) where (x1,y1) is the top-left corner
    and (x2,y2) is the bottom-right corner.

    Returns:
        Float between 0.0 (no overlap) and 1.0 (identical boxes).
    """
    # The intersection rectangle
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # If there's no overlap, intersection area is 0
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    # Union = sum of both areas minus the overlap (otherwise counted twice)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class VehicleTracker:
    """Greedy IoU-based multi-object tracker.

    "Greedy" means we match detections to tracks by picking the highest
    IoU pair first, then the next highest, and so on.  This is simple
    and fast — good enough for the low object counts we see from a
    dashcam.

    Args:
        iou_threshold:    Minimum IoU to consider a match.
        max_disappeared:  Delete a track after this many missed frames.
        max_tracks:       Upper limit on simultaneous tracks (memory guard).
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_disappeared: int = 30,
        max_tracks: int = 50,
    ):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.max_tracks = max_tracks

        # Dict mapping track_id → Track object
        self.tracks: Dict[int, Track] = {}

        # Counter for generating unique IDs (never reused during a session)
        self._next_id = 0

    def _new_id(self) -> int:
        """Generate and return the next unique track ID."""
        tid = self._next_id
        self._next_id += 1
        return tid

    # ------------------------------------------------------------------ #
    #  Main entry point — call this once per frame
    # ------------------------------------------------------------------ #

    def update(
        self, detections: List[Tuple[BBox, float]]
    ) -> Dict[int, Track]:
        """Feed detections from the current frame and update all tracks.

        Args:
            detections: List of (bounding_box, confidence) from the detector.

        Returns:
            The current tracks dictionary (track_id → Track).
        """
        now = time.time()

        # -- No detections this frame: age all existing tracks ---------- #
        if not detections:
            self._age_all()
            return self.tracks

        # -- No existing tracks: create one per detection --------------- #
        if not self.tracks:
            for bbox, conf in detections:
                t = self._new_id()
                self.tracks[t] = Track(track_id=t, bbox=bbox, confidence=conf)
            return self.tracks

        # -- Match detections to existing tracks using IoU -------------- #
        tids = list(self.tracks.keys())                   # e.g. [0, 1, 2]
        t_boxes = [self.tracks[t].bbox for t in tids]     # their bboxes

        # Build an IoU matrix: rows = existing tracks, cols = new detections
        iou_mat = np.zeros((len(tids), len(detections)), dtype=np.float32)
        for i, tb in enumerate(t_boxes):
            for j, (db, _) in enumerate(detections):
                iou_mat[i, j] = _iou(tb, db)

        # Greedy matching: repeatedly pick the highest IoU cell
        matched_t, matched_d = set(), set()
        for _ in range(min(len(tids), len(detections))):
            # Find the cell with the maximum IoU value
            idx = int(np.argmax(iou_mat))
            # Convert flat index back to (row, col)
            i, j = divmod(idx, iou_mat.shape[1])

            if iou_mat[i, j] < self.iou_threshold:
                break  # no more good matches

            # Update the matched track with the new detection's data
            tid = tids[i]
            bbox, conf = detections[j]
            trk = self.tracks[tid]
            trk.bbox = bbox
            trk.confidence = conf
            trk.frames_seen += 1
            trk.frames_missing = 0
            trk.last_seen = now

            matched_t.add(i)
            matched_d.add(j)

            # Zero out the matched row + column so they can't be picked again
            iou_mat[i, :] = 0
            iou_mat[:, j] = 0

        # -- Age unmatched tracks --------------------------------------- #
        to_del = []
        for i, tid in enumerate(tids):
            if i not in matched_t:
                self.tracks[tid].frames_missing += 1
                if self.tracks[tid].frames_missing > self.max_disappeared:
                    to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        # -- Create new tracks for unmatched detections ----------------- #
        for j, (bbox, conf) in enumerate(detections):
            if j not in matched_d and len(self.tracks) < self.max_tracks:
                t = self._new_id()
                self.tracks[t] = Track(track_id=t, bbox=bbox, confidence=conf)

        return self.tracks

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _age_all(self):
        """Increment frames_missing on every track; delete stale ones."""
        to_del = []
        for tid, trk in self.tracks.items():
            trk.frames_missing += 1
            if trk.frames_missing > self.max_disappeared:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

    # ------------------------------------------------------------------ #
    #  Query helpers
    # ------------------------------------------------------------------ #

    def get_ocr_candidates(self, min_frames: int = 5) -> List[Track]:
        """Return tracks that are stable enough for OCR and haven't been read yet.

        "Stable" means:
          - Seen in at least *min_frames* frames.
          - Currently visible (frames_missing == 0).
          - OCR has not already been attempted.
        """
        return [
            t
            for t in self.tracks.values()
            if t.frames_seen >= min_frames
            and t.frames_missing == 0
            and not t.ocr_attempted
        ]
