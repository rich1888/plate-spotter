"""
ocr.py — Tesseract-based OCR for number plates with preprocessing.

Why Tesseract instead of EasyOCR or PaddleOCR?
───────────────────────────────────────────────
  EasyOCR loads ~500 MB of deep-learning models into RAM and takes 5–10 s
  to start up.  PaddleOCR is lighter but still heavy.

  Tesseract is a traditional OCR engine that uses ~50 MB of RAM and starts
  in under a second.  With proper image preprocessing (resize, denoise,
  contrast enhancement, thresholding) it reads UK plates reliably when the
  plate crop is at least ~40 px tall.

Preprocessing pipeline
──────────────────────
  1. Resize to 60 px tall (standardises character size for Tesseract).
  2. Convert to greyscale.
  3. Bilateral filter (reduces noise but keeps edges sharp).
  4. CLAHE (local contrast enhancement — helps under uneven lighting).
  5. Adaptive threshold (converts to black/white — what Tesseract wants).
  6. Add white border (Tesseract struggles when characters touch the edge).
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlateOCR:
    """Pre-process a plate crop and run Tesseract OCR.

    Args:
        config: Full app config dict — we read the "ocr" section.
    """

    def __init__(self, config: dict):
        cfg = config["ocr"]
        self.conf_thresh: float = cfg["confidence_threshold"]  # 0-100
        self.max_blur: float = cfg["max_blur_variance"]        # Laplacian var threshold
        self.tess_cfg: str = cfg["tesseract_config"]           # e.g. "--psm 7 --oem 3"
        self.whitelist: str = cfg["char_whitelist"]            # allowed characters
        self._tess = None  # lazy-loaded below

    def _pytess(self):
        """Lazy-import pytesseract.

        We delay the import so the module can be loaded on machines
        without Tesseract installed (e.g. for running unit tests on the
        preprocessing code).
        """
        if self._tess is None:
            import pytesseract
            self._tess = pytesseract
        return self._tess

    # ------------------------------------------------------------------ #
    #  Blur detection
    # ------------------------------------------------------------------ #

    @staticmethod
    def blur_score(img: np.ndarray) -> float:
        """Measure image sharpness using the Laplacian variance.

        The Laplacian highlights edges.  A sharp image has many strong
        edges → high variance.  A blurry image has few → low variance.

        Args:
            img: Greyscale or BGR image.

        Returns:
            Variance (float).  Higher = sharper.  Typical threshold ~100.
        """
        g = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    # ------------------------------------------------------------------ #
    #  Image preprocessing
    # ------------------------------------------------------------------ #

    @staticmethod
    def preprocess(plate: np.ndarray) -> np.ndarray:
        """Prepare a plate crop for Tesseract.

        Steps:
          1. Resize height to 60 px (keeps characters consistently sized).
          2. Convert to greyscale.
          3. Bilateral filter — smooths noise, preserves character edges.
          4. CLAHE — improves contrast locally (handles shadows / glare).
          5. Adaptive threshold — clean black text on white background.
          6. White border — gives Tesseract padding around the characters.

        Args:
            plate: BGR plate crop (numpy array).

        Returns:
            Binary (black & white) image ready for Tesseract.
        """
        if plate is None or plate.size == 0:
            return plate

        img = plate.copy()

        # -- Step 1: normalise height to 60 px --
        h, w = img.shape[:2]
        if h > 0:
            scale = 60 / h
            img = cv2.resize(img, (int(w * scale), 60), interpolation=cv2.INTER_CUBIC)

        # -- Step 2: greyscale --
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # -- Step 3: bilateral filter (denoise) --
        # d=9: neighbourhood diameter.  sigmaColor / sigmaSpace = 75.
        dn = cv2.bilateralFilter(gray, 9, 75, 75)

        # -- Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalisation) --
        # Divides image into 8×8 tiles; equalises histogram in each tile.
        # clipLimit=2.0 prevents over-amplification of noise.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(dn)

        # -- Step 5: adaptive threshold --
        # Converts to pure black & white.  "GAUSSIAN_C" means the threshold
        # for each pixel is a Gaussian-weighted average of nearby pixels,
        # which handles uneven lighting well.
        th = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # block size (neighbourhood radius in pixels)
            2,   # constant subtracted from mean
        )

        # -- Step 5b: handle inverted plates (white text on dark background) --
        # If the image is mostly white after thresholding, it's a normal plate.
        # If mostly black, it's inverted — flip it so Tesseract sees dark text
        # on a white background (which is what it expects).
        white_ratio = np.count_nonzero(th) / th.size
        if white_ratio < 0.3:
            th = cv2.bitwise_not(th)

        # -- Step 6: white border --
        # Tesseract can misread characters that touch the image edge.
        # Adding 10 px of white padding on all sides fixes this.
        return cv2.copyMakeBorder(th, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

    # ------------------------------------------------------------------ #
    #  Main OCR entry point
    # ------------------------------------------------------------------ #

    def read(self, plate_img: np.ndarray) -> Optional[Tuple[str, float]]:
        """OCR a plate crop image.

        Checks blur first, preprocesses, runs Tesseract, filters by
        confidence.

        Args:
            plate_img: BGR plate crop (numpy array).

        Returns:
            (text, average_confidence) tuple, or None if the image was
            too blurry, OCR failed, or confidence was below threshold.
        """
        if plate_img is None or plate_img.size == 0:
            return None

        # Reject blurry images before wasting time on OCR
        bscore = self.blur_score(plate_img)
        if bscore < self.max_blur:
            logger.debug("Plate too blurry (var=%.1f)", bscore)
            return None

        # Pre-process into binary image
        processed = self.preprocess(plate_img)

        # Build Tesseract config string with whitelist
        tess = self._pytess()
        cfg = f"{self.tess_cfg} -c tessedit_char_whitelist={self.whitelist}"

        try:
            # image_to_data returns per-word text, confidence, and position
            data = tess.image_to_data(
                processed, config=cfg, output_type=tess.Output.DICT
            )
        except Exception as exc:
            logger.error("Tesseract error: %s", exc)
            return None

        # Collect non-empty text segments and their confidences
        parts, confs = [], []
        for txt, c in zip(data["text"], data["conf"]):
            t = txt.strip()
            c = int(c)
            if t and c > 0:  # skip empty strings and zero-confidence entries
                parts.append(t)
                confs.append(c)

        if not parts:
            return None

        # Join all text segments into one string (plates are a single line)
        full = "".join(parts)
        avg = sum(confs) / len(confs)

        # Reject low-confidence results
        if avg < self.conf_thresh:
            logger.debug("Low OCR confidence %.1f  text=%s", avg, full)
            return None

        return full, avg
