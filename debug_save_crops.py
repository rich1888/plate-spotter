#!/usr/bin/env python3
"""Save the plate detection crops for visual debugging."""
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detect import VehicleDetector, PlateLocalizer
from src.ocr import PlateOCR

cfg = {
    "detection": {
        "model_path": "yolov8n.pt",
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

det = VehicleDetector(cfg)
det.load()
dets = det.detect(img)
print(f"Vehicles: {len(dets)}")

if not dets:
    print("No vehicles found")
    sys.exit(0)

ploc = PlateLocalizer(cfg)
x1, y1, x2, y2 = dets[0][0]
roi = img[y1:y2, x1:x2]
result = ploc.find(roi)

cv2.imwrite("/home/pi/debug_roi.jpg", roi)
print("Saved: /home/pi/debug_roi.jpg (vehicle crop)")

if result is None:
    print("No plate region found")
    sys.exit(0)

plate_img, (px, py, pw, ph) = result
print(f"Plate region: ({px},{py},{pw},{ph}) size={pw}x{ph}px")

processed = PlateOCR.preprocess(plate_img)
cv2.imwrite("/home/pi/debug_plate.jpg", plate_img)
cv2.imwrite("/home/pi/debug_processed.jpg", processed)
print("Saved: /home/pi/debug_plate.jpg (plate crop)")
print("Saved: /home/pi/debug_processed.jpg (after OCR preprocessing)")
print()
print("Now copy them to your Mac:")
print("  scp pi@<pi-ip>:/home/pi/debug_*.jpg ~/Desktop/")
