#!/usr/bin/env python3
"""Debug script to see raw Tesseract output on a test image."""
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detect import VehicleDetector, PlateLocalizer
from src.ocr import PlateOCR
import pytesseract

cfg = {
    "detection": {
        "model_path": "yolov8n.onnx",
        "confidence": 0.2,
        "vehicle_classes": [2, 3, 5, 7],
        "input_size": 640,
        "detection_interval": 3,
    },
    "plate_detection": {
        "min_aspect_ratio": 2.0,
        "max_aspect_ratio": 7.0,
        "min_area_ratio": 0.005,
        "max_area_ratio": 0.15,
        "min_plate_height_px": 20,
    },
    "ocr": {
        "confidence_threshold": 10,
        "max_blur_variance": 100,
        "tesseract_config": "--psm 7 --oem 3",
        "char_whitelist": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    },
}

image_path = sys.argv[1] if len(sys.argv) > 1 else "/home/pi/test_reg.jpg"
img = cv2.imread(image_path)
if img is None:
    print(f"ERROR: Could not read {image_path}")
    sys.exit(1)

print(f"Image: {image_path} ({img.shape[1]}x{img.shape[0]})")

det = VehicleDetector(cfg)
det.load()
dets = det.detect(img)
print(f"Vehicles: {len(dets)}")

ploc = PlateLocalizer(cfg)
for i, (bbox, conf) in enumerate(dets):
    x1, y1, x2, y2 = bbox
    print(f"\nVehicle {i+1}: bbox=({x1},{y1},{x2},{y2}) conf={conf:.2f}")
    roi = img[y1:y2, x1:x2]
    result = ploc.find(roi)
    if result:
        plate_img, (px, py, pw, ph) = result
        print(f"  Plate region: {pw}x{ph}px")
        blur = PlateOCR.blur_score(plate_img)
        print(f"  Blur score: {blur:.1f}")
        processed = PlateOCR.preprocess(plate_img)
        raw = pytesseract.image_to_data(
            processed,
            config="--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            output_type=pytesseract.Output.DICT,
        )
        print(f"  Raw Tesseract output:")
        for t, c in zip(raw["text"], raw["conf"]):
            if t.strip():
                print(f"    text=\"{t.strip()}\"  conf={c}")
        if not any(t.strip() for t in raw["text"]):
            print(f"    (empty - Tesseract returned nothing)")
    else:
        print("  No plate region found")
