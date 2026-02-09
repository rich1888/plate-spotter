#!/usr/bin/env python3
"""
test_with_image.py — Test the plate-spotter pipeline on static images.

Use this on your laptop/desktop to check accuracy before deploying to the Pi.
No camera or Pi hardware needed.

Usage:
    # Single image:
    python3 test_with_image.py photo_of_car.jpg

    # Multiple images:
    python3 test_with_image.py img1.jpg img2.png img3.jpg

    # A whole folder:
    python3 test_with_image.py ./test_photos/

    # Show the detection visually (opens a window — needs a display):
    python3 test_with_image.py --show photo_of_car.jpg

    # Save annotated output images:
    python3 test_with_image.py --save-output ./results/ photo_of_car.jpg

What it does:
    1. Loads each image from disk (no camera).
    2. Runs YOLOv8n vehicle detection.
    3. For each vehicle, runs morphological plate localisation.
    4. Runs Tesseract OCR on the plate crop.
    5. Validates the result as a UK plate.
    6. Prints results to the terminal.

Requirements:
    pip3 install opencv-python-headless numpy PyYAML ultralytics pytesseract
    # Plus Tesseract must be installed:
    #   macOS:   brew install tesseract
    #   Ubuntu:  sudo apt install tesseract-ocr
    #   Windows: download from https://github.com/UB-Mannheim/tesseract/wiki
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Add the project root to Python's module search path so we can import src.*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.detect import VehicleDetector, PlateLocalizer
from src.ocr import PlateOCR
from src.uk_plate import classify_plate, normalize_plate


def collect_images(paths: list) -> list:
    """Expand paths — if a directory, grab all image files inside it."""
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in IMAGE_EXTS:
                    files.append(str(f))
        elif p.is_file():
            files.append(str(p))
        else:
            print(f"  WARNING: '{p}' not found, skipping")
    return files


def process_image(
    img_path: str,
    detector: VehicleDetector,
    plate_loc: PlateLocalizer,
    ocr: PlateOCR,
    config: dict,
    show: bool = False,
    save_dir: str = None,
):
    """Run the full pipeline on a single image file."""
    print(f"\n{'='*60}")
    print(f"  Image: {img_path}")
    print(f"{'='*60}")

    # Load the image from disk
    frame = cv2.imread(img_path)
    if frame is None:
        print("  ERROR: Could not read image")
        return

    h, w = frame.shape[:2]
    print(f"  Size: {w}x{h}")

    # -- Step 1: Detect vehicles ----------------------------------------
    detections = detector.detect(frame)
    print(f"  Vehicles found: {len(detections)}")

    if not detections:
        print("  No vehicles detected.")
        return

    # For visual output, draw on a copy of the frame
    annotated = frame.copy() if (show or save_dir) else None

    # -- Step 2–5: For each vehicle, find plate → OCR → validate --------
    for i, (bbox, vconf) in enumerate(detections):
        x1, y1, x2, y2 = bbox
        print(f"\n  Vehicle {i+1}:  bbox=({x1},{y1},{x2},{y2})  conf={vconf:.2f}")

        # Crop the vehicle region
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            print("    Skipped (empty ROI)")
            continue

        # Draw vehicle box on annotated image
        if annotated is not None:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Find plate in vehicle ROI
        result = plate_loc.find(roi)
        if result is None:
            print("    No plate-like region found")
            continue

        plate_img, (px, py, pw, ph) = result
        print(f"    Plate region: ({px},{py},{pw},{ph})  size={pw}x{ph}px")

        # Draw plate box on annotated image (offset to full-frame coords)
        if annotated is not None:
            cv2.rectangle(
                annotated,
                (x1 + px, y1 + py),
                (x1 + px + pw, y1 + py + ph),
                (0, 0, 255), 2,
            )

        # Check blur
        blur = ocr.blur_score(plate_img)
        print(f"    Blur score: {blur:.1f}  (threshold: {ocr.max_blur})")

        # Run OCR
        ocr_result = ocr.read(plate_img)
        if ocr_result is None:
            print("    OCR failed (too blurry or low confidence)")
            continue

        text, conf = ocr_result
        norm = normalize_plate(text)
        print(f"    OCR raw:    '{text}'")
        print(f"    Normalised: '{norm}'")
        print(f"    Confidence: {conf:.1f}")

        # UK plate validation
        info = classify_plate(
            text,
            min_length=config["uk_plate"]["min_length"],
            max_length=config["uk_plate"]["max_length"],
            reject_all_same=config["uk_plate"]["reject_all_same"],
        )
        if info is None:
            print("    UK validation: REJECTED (not a plausible UK plate)")
        else:
            print(f"    UK validation: PASS")
            print(f"    Format:        {info['format']}")
            print(f"    Dateless:      {info['dateless']}")

            # Draw text on annotated image
            if annotated is not None:
                label = f"{norm} ({info['format']})"
                cv2.putText(
                    annotated, label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )

    # -- Show / save annotated image ------------------------------------
    if annotated is not None and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "result_" + os.path.basename(img_path))
        cv2.imwrite(out_path, annotated)
        print(f"\n  Saved annotated image: {out_path}")

    if annotated is not None and show:
        cv2.imshow("plate-spotter", annotated)
        print("\n  Press any key to continue (or 'q' to quit)…")
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        description="Test plate-spotter on static images (no Pi needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 test_with_image.py car_photo.jpg
  python3 test_with_image.py --show ./test_photos/
  python3 test_with_image.py --save-output ./results/ img1.jpg img2.jpg
        """,
    )
    ap.add_argument("images", nargs="+", help="Image files or directories")
    ap.add_argument("-c", "--config", default=None, help="Path to config.yaml")
    ap.add_argument("--show", action="store_true",
                    help="Show annotated images in a window")
    ap.add_argument("--save-output", default=None, metavar="DIR",
                    help="Save annotated result images to this directory")
    args = ap.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialise pipeline components
    print("Loading YOLO model (first run downloads ~6 MB)…")
    detector = VehicleDetector(config)
    detector.load()

    plate_loc = PlateLocalizer(config)
    ocr = PlateOCR(config)

    # Collect all image files
    image_files = collect_images(args.images)
    if not image_files:
        print("No image files found!")
        sys.exit(1)

    print(f"Processing {len(image_files)} image(s)…")

    for img_path in image_files:
        process_image(
            img_path, detector, plate_loc, ocr, config,
            show=args.show, save_dir=args.save_output,
        )

    if args.show:
        cv2.destroyAllWindows()

    print(f"\nDone — processed {len(image_files)} image(s).")


if __name__ == "__main__":
    main()
