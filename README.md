# plate-spotter

Raspberry Pi vehicle plate recognition system. Captures passing vehicles via
the Pi camera, detects plates, OCRs them, validates against UK formats,
identifies dateless plates, and uploads "spotted" events to the RegHistory API.

---

## How it works (the short version)

```
Camera  →  detect vehicles (YOLO)  →  find plate in vehicle (OpenCV)
        →  read plate text (Tesseract OCR)  →  validate as UK plate
        →  save evidence  →  upload to API
```

The whole thing runs as a background service on your Pi. It auto-starts on
boot and logs everything to a file.

---

## Testing on your laptop (no Pi needed)

You can test the detection + OCR accuracy on your Mac/PC using photos of
cars. No camera or Pi hardware required.

### 1. Install Python dependencies

**macOS** (your current setup):

```bash
# Install Homebrew if you don't have it:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Tesseract OCR engine:
brew install tesseract

# Go to the project:
cd ~/code/plate-spotter

# Create a virtual environment (like composer's vendor/ — keeps deps isolated):
python3 -m venv venv

# Activate it (you'll see "(venv)" in your prompt):
source venv/bin/activate

# Install Python packages:
pip install -r requirements.txt
```

**Windows**:
```
# Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
# Then in PowerShell:
cd plate-spotter
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the unit tests

```bash
# Make sure venv is activated (you see "(venv)" in your prompt)
source venv/bin/activate

# Run all tests (no camera/Tesseract needed for these):
python3 -m pytest tests/ -v
```

You should see all 51 tests pass.

### 3. Test with real photos

Save some photos of cars with visible number plates (Google Images works,
or take photos with your phone).

```bash
# Test a single image:
python3 test_with_image.py photo_of_car.jpg

# Test a folder of images:
python3 test_with_image.py ./test_photos/

# Save annotated result images (boxes drawn around vehicles + plates):
python3 test_with_image.py --save-output ./results/ photo_of_car.jpg

# Show results in a popup window:
python3 test_with_image.py --show photo_of_car.jpg
```

The script will print what it finds for each image:

```
============================================================
  Image: photo_of_car.jpg
============================================================
  Size: 1280x720
  Vehicles found: 1

  Vehicle 1:  bbox=(120,200,580,520)  conf=0.87
    Plate region: (45,180,210,48)  size=210x48px
    Blur score: 342.5  (threshold: 100.0)
    OCR raw:    'AB12 CDE'
    Normalised: 'AB12CDE'
    Confidence: 78.3
    UK validation: PASS
    Format:        current
    Dateless:      False
```

### 4. Deactivate when done

```bash
deactivate   # exits the virtual environment
```

---

## Installing on a Raspberry Pi (fresh OS)

These instructions assume you're starting from a **fresh Raspberry Pi OS
(Trixie, Bookworm, or Bullseye)** install with the Pi Camera Module connected.

### Prerequisites

- Raspberry Pi 4 (4 GB+) or Pi 5
- Raspberry Pi Camera Module (v2 or HQ Camera)
- microSD card with Raspberry Pi OS installed
- Internet connection (for downloading packages)

### Step 1: Initial Pi setup

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Enable the camera
# On Bullseye you may need: sudo raspi-config → Interface Options → Camera
# Test the camera works:
rpicam-hello --timeout 3000
```

If you see a 3-second camera preview, you're good.

### Step 2: Install system packages

```bash
sudo apt install -y \
    git \
    python3-venv python3-dev python3-pip \
    tesseract-ocr libtesseract-dev libleptonica-dev \
    libcamera-dev libcap-dev python3-libcamera python3-picamera2 \
    libopencv-dev python3-opencv \
    libopenblas-dev
```

What these are:
- `git` — to clone the repo
- `python3-venv` — creates isolated Python environments (like composer)
- `tesseract-ocr` — the OCR engine that reads text from images
- `python3-picamera2` — Python library to control the Pi camera
- `python3-opencv` — computer vision library (image processing)
- `libopenblas-dev` — fast maths library that numpy needs

### Step 3: Clone and install

```bash
# Clone the repo
git clone <YOUR_REPO_URL> ~/plate-spotter
cd ~/plate-spotter

# Create virtual environment
# --system-site-packages lets us use the apt-installed picamera2 + opencv
python3 -m venv --system-site-packages venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip wheel
pip install -r requirements.txt

# Copy the example config
cp config.example.yaml config.yaml
```

### Step 4: Configure

Edit `config.yaml` with your settings:

```bash
nano config.yaml
```

Key things to set:

```yaml
upload:
  api_token: "your-actual-token-here"   # or use env var instead
  device_id: "pi1"                       # name for this Pi

camera:
  use_picamera2: true                    # true for Pi camera
  resolution: [1280, 720]               # lower = faster, higher = better OCR
```

**Or** set the token as an environment variable (more secure):

```bash
export PLATE_SPOTTER_API_TOKEN="your-actual-token-here"
```

### Step 5: Test it works

```bash
# Quick smoke test — run for a few seconds and check for errors:
source venv/bin/activate
python3 -m src.main -c config.yaml

# You should see:
#   plate-spotter starting
#   ONNX model loaded and warmed up
#   Camera started
#   Uploader thread started
#
# Press Ctrl-C to stop
```

### Step 6: Install as a system service

This makes plate-spotter start automatically on boot:

```bash
# Edit the service file to match your paths/user if needed:
nano plate-spotter.service

# Install it:
sudo cp plate-spotter.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable plate-spotter
sudo systemctl start plate-spotter

# Check it's running:
sudo systemctl status plate-spotter

# Watch the logs in real time:
sudo journalctl -u plate-spotter -f
```

To stop/restart:

```bash
sudo systemctl stop plate-spotter
sudo systemctl restart plate-spotter
```

### One-liner install (alternative)

If you prefer, there's an install script that does steps 2–6 automatically:

```bash
cd ~/plate-spotter
chmod +x install.sh
./install.sh /home/pi/plate-spotter
```

**You still need to edit `config.yaml` afterwards to set your API token.**

---

## Architecture

```
┌──────────┐     ┌─────────────────────────────────┐     ┌───────────┐
│  Camera   │────▶│  Processing loop (main thread)  │────▶│  Upload   │
│  thread   │     │                                 │     │  thread   │
│           │     │  1. YOLO vehicle detect          │     │           │
│  picamera2│     │     (every 3rd frame)            │     │  SQLite   │
│  15 fps   │     │  2. IoU tracker                  │     │  queue    │
│           │     │     (every frame)                │     │     ↓     │
│  frame    │     │  3. Plate localisation (OpenCV)  │     │  HTTP POST│
│  queue    │     │  4. OCR (Tesseract)              │     │  to API   │
│  (10 max) │     │  5. UK plate validation          │     │  w/ retry │
└──────────┘     │  6. Save evidence + enqueue       │     └───────────┘
                  └─────────────────────────────────┘
```

### Design choices

| Decision | Choice | Why |
|---|---|---|
| Vehicle detection | YOLOv8n via ONNX Runtime | ~100 MB RAM vs ~800 MB with PyTorch |
| Plate localisation | OpenCV morphological (blackhat + edge) | No second model needed, < 1 ms per ROI |
| OCR | Tesseract via pytesseract | ~50 MB RAM vs 500 MB+ for EasyOCR/PaddleOCR |
| Tracking | Greedy IoU tracker | No external deps, handles dashcam motion well |
| Upload queue | SQLite WAL | Crash-safe, survives reboots, no extra service |
| Frame skipping | Detection every 3rd frame | Keeps CPU headroom for OCR + upload |

---

## Configuration reference

All settings live in `config.yaml`. Here are the most useful tunables:

| Setting | Default | What it does |
|---|---|---|
| `camera.resolution` | [1280, 720] | Camera resolution. Lower = faster. |
| `camera.fps` | 15 | Frame capture rate |
| `camera.use_picamera2` | true | false = use USB webcam (for testing) |
| `detection.detection_interval` | 3 | Run YOLO every N-th frame |
| `detection.input_size` | 640 | YOLO input size. 320 = ~2x faster. |
| `detection.confidence` | 0.4 | Min YOLO confidence (0–1) |
| `tracking.min_stable_frames` | 5 | Frames before OCR fires |
| `ocr.confidence_threshold` | 50 | Reject OCR below this (0–100) |
| `ocr.max_blur_variance` | 100 | Below = too blurry to OCR |
| `dedupe.window_minutes` | 5 | Ignore same plate for N minutes |
| `storage.max_size_mb` | 500 | Auto-delete oldest evidence over this |
| `storage.max_age_days` | 7 | Auto-delete evidence older than this |
| `upload.api_token` | (empty) | Your RegHistory API token |
| `upload.device_id` | pi1 | Name for this Pi |

---

## Dateless plate rule

A plate is **dateless** when its normalised string starts **or** ends with
a digit:

| Plate | Dateless? | Why |
|---|---|---|
| AB12CDE | No | Starts A, ends E |
| 1234ABC | Yes | Starts with 1 |
| ABC1234 | Yes | Ends with 4 |
| A1 | Yes | Ends with 1 |

---

## Project structure

```
plate-spotter/
├── src/
│   ├── __init__.py         ← makes src/ a Python package
│   ├── camera.py           ← picamera2 capture thread (OpenCV fallback)
│   ├── config.py           ← YAML config with defaults + env overrides
│   ├── detect.py           ← YOLOv8n (ONNX) vehicles + morphological plate finder
│   ├── main.py             ← pipeline orchestrator + CLI entry point
│   ├── ocr.py              ← Tesseract OCR with preprocessing
│   ├── queue_manager.py    ← SQLite-backed persistent upload queue
│   ├── storage.py          ← rotating evidence storage (images + JSON)
│   ├── track.py            ← IoU-based vehicle tracker
│   ├── uk_plate.py         ← UK plate validation + dateless detection
│   └── uploader.py         ← API upload + deduplication
├── tests/
│   ├── test_uk_plate.py    ← 46 unit tests for plate validation
│   └── test_ocr.py         ← 5 preprocessing smoke tests
├── test_with_image.py      ← test on static images (no Pi needed)
├── config.example.yaml     ← copy to config.yaml and edit
├── requirements.txt        ← Python dependencies
├── plate-spotter.service   ← systemd unit file (auto-start on boot)
├── install.sh              ← one-shot Pi installer script
└── README.md               ← this file
```

---

## Troubleshooting

**"No module named picamera2"** — You're on a laptop, not a Pi. Set
`camera.use_picamera2: false` in config.yaml to use a webcam instead.

**"tesseract is not installed"** — Install it:
- macOS: `brew install tesseract`
- Pi/Ubuntu: `sudo apt install tesseract-ocr`

**"No vehicles detected"** — Try lowering `detection.confidence` to 0.25.
Or check that the image actually contains a car.

**Camera permission error on Pi** — Add your user to the video group:
`sudo usermod -aG video $USER` then log out and back in.

**ONNX model not found** — You need to export it once (requires ultralytics):
```bash
pip install ultralytics
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx')"
pip uninstall ultralytics torch -y  # optional: reclaim disk/RAM
```
This creates `yolov8n.onnx` in the project directory.
