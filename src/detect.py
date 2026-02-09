"""
detect.py — Vehicle detection (YOLOv8n via ONNX Runtime) + morphological
plate localisation.

Two-stage pipeline:
  1. VehicleDetector  — runs YOLOv8-nano (ONNX) on the full frame to find
     cars, trucks, buses, motorcycles.  Returns bounding boxes.
  2. PlateLocalizer   — runs *inside* each vehicle bounding box (ROI) using
     cheap OpenCV morphological operations (no second neural network).
     Returns the cropped plate image and its coordinates.

Why this split?
  - Running YOLO once per frame is expensive enough on a Pi.
  - The morphological plate finder costs < 1 ms and needs no GPU, making
    it essentially free by comparison.
  - Searching for a plate only inside the vehicle box (instead of the whole
    frame) dramatically reduces false positives.

COCO class IDs used:
  2 = car,  3 = motorcycle,  5 = bus,  7 = truck
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Type alias — a bounding box is four integers: (x1, y1, x2, y2)
BBox = Tuple[int, int, int, int]


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 — Vehicle detection
# ═══════════════════════════════════════════════════════════════════════════

class VehicleDetector:
    """Detect vehicles in a frame using YOLOv8-nano (ONNX Runtime).

    Requires a pre-exported ``yolov8n.onnx`` model file.  Export once on
    the Pi with ultralytics installed::

        python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"

    Args:
        config: Full app config dict — we read the "detection" section.
    """

    # NMS IoU threshold for overlapping boxes
    _NMS_IOU = 0.45

    def __init__(self, config: dict):
        cfg = config["detection"]
        self.model_path: str = cfg["model_path"]   # e.g. "yolov8n.onnx"
        self.conf: float = cfg["confidence"]        # minimum confidence (0–1)
        self.classes: set = set(cfg["vehicle_classes"])  # COCO IDs to keep
        self.imgsz: int = cfg["input_size"]          # inference resolution
        self._session = None  # loaded lazily in load()

    def load(self):
        """Load the ONNX model into an inference session.

        Also runs a single "warm-up" inference on a blank image so the
        first real frame isn't artificially slow.
        """
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )

        # Warm-up: a dummy black image at the target resolution
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._preprocess_and_infer(dummy)
        logger.info("ONNX model loaded and warmed up: %s", self.model_path)

    # ------------------------------------------------------------------ #
    #  Preprocessing
    # ------------------------------------------------------------------ #

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """Resize + normalise a BGR frame for YOLO input.

        Uses letter-boxing (pad with grey) to maintain aspect ratio.

        Returns:
            (blob, ratio, pad_x, pad_y) — the NCHW float32 tensor,
            the resize ratio, and the x/y padding offsets.
        """
        h, w = frame.shape[:2]
        ratio = self.imgsz / max(h, w)
        new_w, new_h = int(w * ratio), int(h * ratio)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square (imgsz × imgsz), grey fill
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        padded = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # HWC → CHW, BGR → RGB, 0-255 → 0-1, add batch dim
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]  # (1, 3, 640, 640)

        return blob, ratio, pad_x, pad_y

    # ------------------------------------------------------------------ #
    #  Postprocessing
    # ------------------------------------------------------------------ #

    def _postprocess(
        self, output: np.ndarray, ratio: float, pad_x: int, pad_y: int,
        orig_h: int, orig_w: int,
    ) -> List[Tuple[BBox, float]]:
        """Parse raw YOLO output tensor and apply NMS.

        Args:
            output: shape (1, 84, 8400) — 4 box coords + 80 class scores
                    per candidate.
            ratio:  scale factor from preprocess.
            pad_x, pad_y: letter-box padding offsets.
            orig_h, orig_w: original frame dimensions.

        Returns:
            List of ((x1, y1, x2, y2), confidence) for vehicle detections.
        """
        # Transpose to (8400, 84) — one row per candidate
        preds = output[0].T  # (8400, 84)

        # Columns: cx, cy, w, h, then 80 class scores
        cx = preds[:, 0]
        cy = preds[:, 1]
        bw = preds[:, 2]
        bh = preds[:, 3]
        class_scores = preds[:, 4:]  # (8400, 80)

        # Best class per candidate
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Filter: confidence threshold AND vehicle class
        mask = confidences >= self.conf
        class_mask = np.isin(class_ids, list(self.classes))
        keep = mask & class_mask

        cx, cy, bw, bh = cx[keep], cy[keep], bw[keep], bh[keep]
        confidences = confidences[keep]

        if len(confidences) == 0:
            return []

        # Convert centre-wh to x1y1x2y2 (still in padded 640×640 space)
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Remove letter-box padding and rescale to original frame
        x1 = (x1 - pad_x) / ratio
        y1 = (y1 - pad_y) / ratio
        x2 = (x2 - pad_x) / ratio
        y2 = (y2 - pad_y) / ratio

        # NMS via OpenCV
        boxes_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            confidences.tolist(),
            self.conf,
            self._NMS_IOU,
        )
        if len(indices) == 0:
            return []

        out: List[Tuple[BBox, float]] = []
        for idx in indices:
            i = int(idx)
            bx1 = max(0, int(x1[i]))
            by1 = max(0, int(y1[i]))
            bx2 = min(orig_w, int(x2[i]))
            by2 = min(orig_h, int(y2[i]))
            out.append(((bx1, by1, bx2, by2), float(confidences[i])))

        return out

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def _preprocess_and_infer(self, frame: np.ndarray):
        """Run preprocessing + ONNX inference, return raw outputs + geometry."""
        blob, ratio, pad_x, pad_y = self._preprocess(frame)
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: blob})
        return outputs[0], ratio, pad_x, pad_y

    def detect(self, frame: np.ndarray) -> List[Tuple[BBox, float]]:
        """Run vehicle detection on *frame*.

        Args:
            frame: BGR image (numpy array, shape H×W×3).

        Returns:
            List of ((x1, y1, x2, y2), confidence) tuples in pixel coords
            of the original frame.
        """
        if self._session is None:
            self.load()

        h, w = frame.shape[:2]
        output, ratio, pad_x, pad_y = self._preprocess_and_infer(frame)
        return self._postprocess(output, ratio, pad_x, pad_y, h, w)


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 — Plate localisation (no neural network)
# ═══════════════════════════════════════════════════════════════════════════

class PlateLocalizer:
    """Find a number-plate rectangle inside a vehicle crop using OpenCV.

    Uses two complementary methods in parallel:
      - Morphological "blackhat" filtering (highlights dark text on light plate)
      - Sobel edge detection (highlights strong horizontal edges)

    Both methods find contours, filter by aspect ratio and area, and score
    candidates.  The best-scoring candidate wins.

    Why not a second neural network?
      - On a Pi 4, a second YOLO inference would double our processing time.
      - The morphological approach is almost free (< 1 ms) and works well
        for UK plates which have a very consistent aspect ratio (~4.7:1).

    Args:
        config: Full app config dict — we read the "plate_detection" section.
    """

    def __init__(self, config: dict):
        cfg = config["plate_detection"]
        self.min_ar: float = cfg["min_aspect_ratio"]   # minimum width/height ratio
        self.max_ar: float = cfg["max_aspect_ratio"]   # maximum width/height ratio
        self.min_area_ratio: float = cfg["min_area_ratio"]  # plate area ÷ ROI area
        self.max_area_ratio: float = cfg["max_area_ratio"]
        self.min_h: int = cfg.get("min_plate_height_px", 20)

    def find(
        self, roi: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Search for the best plate candidate in a vehicle ROI.

        Args:
            roi: BGR image cropped to the vehicle bounding box.

        Returns:
            (plate_crop, (x, y, w, h)) where coordinates are relative
            to the ROI, or None if no plate-like region was found.
        """
        if roi is None or roi.size == 0:
            return None

        rh, rw = roi.shape[:2]
        roi_area = rh * rw
        if roi_area < 200:
            return None  # vehicle box too small to contain a readable plate

        # Convert to greyscale once — both methods need it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Collect candidates from all methods
        candidates = self._morph(gray, rw, rh, roi_area)
        candidates += self._morph_inverted(gray, rw, rh, roi_area)
        candidates += self._edges(gray, rw, rh, roi_area)

        if not candidates:
            return None

        # Pick the highest-scoring candidate
        best_rect, _ = max(candidates, key=lambda c: c[1])
        x, y, cw, ch = best_rect

        # Clamp to ROI bounds
        x, y = max(0, x), max(0, y)
        cw = min(cw, rw - x)
        ch = min(ch, rh - y)

        if ch < self.min_h or cw < 4:
            return None

        crop = roi[y : y + ch, x : x + cw]
        return (crop, (x, y, cw, ch)) if crop.size else None

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _score(self, aspect: float, y: int, rh: int, area_ratio: float) -> float:
        """Score a plate candidate — higher is better.

        Factors:
          - Aspect ratio closeness to 4.7 (ideal UK plate).
          - Vertical position — plates are usually in the lower half.
          - Area ratio — bigger plates are easier to read.
        """
        ideal = 4.7  # UK plate aspect ratio: 520 mm ÷ 111 mm
        ar_s = max(0.0, 1.0 - abs(aspect - ideal) / ideal)
        pos_s = y / rh  # 0 = top, 1 = bottom; prefer bottom
        return ar_s * 0.6 + pos_s * 0.3 + min(area_ratio * 10, 1.0) * 0.1

    def _filter(self, contours, rw, rh, roi_area):
        """Filter contours by aspect ratio and area, return scored candidates."""
        hits = []
        for cnt in contours:
            # cv2.boundingRect returns the smallest upright rectangle
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch == 0:
                continue
            ar = cw / ch                   # aspect ratio
            ar_r = (cw * ch) / roi_area    # area ratio vs full ROI
            if (
                self.min_ar <= ar <= self.max_ar
                and self.min_area_ratio <= ar_r <= self.max_area_ratio
                and ch >= self.min_h
            ):
                hits.append(((x, y, cw, ch), self._score(ar, y, rh, ar_r)))
        return hits

    def _morph(self, gray, rw, rh, roi_area):
        """Morphological blackhat method.

        Blackhat = closing(image) - image.  It highlights small dark
        regions on a lighter background — exactly what plate characters
        look like.  We then close the gaps between characters and look
        for contours with a plate-like shape.
        """
        # Structuring element sized to capture individual characters
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kern)

        # Otsu threshold — automatically picks the best threshold value
        _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Close small gaps between characters to form one plate-shaped blob
        cl = cv2.morphologyEx(
            th,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),
        )

        # Find external contours (outlines of white regions)
        cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._filter(cnts, rw, rh, roi_area)

    def _morph_inverted(self, gray, rw, rh, roi_area):
        """Morphological tophat method for inverted plates.

        Tophat = image - opening(image).  It highlights small light
        regions on a darker background — white text on black plate
        surrounds (common on dateless / show plates).
        """
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        th_img = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)

        _, th = cv2.threshold(th_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cl = cv2.morphologyEx(
            th,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),
        )

        cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._filter(cnts, rw, rh, roi_area)

    def _edges(self, gray, rw, rh, roi_area):
        """Sobel edge-based method.

        Plates have strong vertical edges (character strokes), so the
        horizontal Sobel operator lights them up.  We threshold and close
        just like the morphological method.
        """
        # Bilateral filter smooths noise but preserves edges
        blur = cv2.bilateralFilter(gray, 11, 17, 17)

        # Sobel in the X direction — highlights vertical edges
        sx = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)

        _, th = cv2.threshold(sx, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cl = cv2.morphologyEx(
            th,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)),
        )

        cnts, _ = cv2.findContours(cl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._filter(cnts, rw, rh, roi_area)
