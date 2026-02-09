#!/usr/bin/env bash
# install.sh — one-shot setup for plate-spotter on Raspberry Pi OS
set -euo pipefail

INSTALL_DIR="${1:-/home/pi/plate-spotter}"
VENV="$INSTALL_DIR/venv"

echo "=== plate-spotter installer ==="
echo "Install dir: $INSTALL_DIR"

# ── system packages ──────────────────────────────────────────────────── #
echo ">> Installing system packages …"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv python3-dev \
    tesseract-ocr libtesseract-dev libleptonica-dev \
    libcamera-dev libcap-dev python3-libcamera python3-picamera2 \
    libopencv-dev python3-opencv \
    libatlas-base-dev

# ── copy project ─────────────────────────────────────────────────────── #
if [ "$(realpath .)" != "$(realpath "$INSTALL_DIR")" ]; then
    echo ">> Copying project to $INSTALL_DIR …"
    mkdir -p "$INSTALL_DIR"
    cp -r src tests config.example.yaml requirements.txt plate-spotter.service "$INSTALL_DIR/"
fi

# ── venv + pip ────────────────────────────────────────────────────────── #
echo ">> Creating virtualenv …"
python3 -m venv --system-site-packages "$VENV"
# --system-site-packages lets us use apt-installed picamera2 & opencv

echo ">> Installing Python dependencies …"
"$VENV/bin/pip" install --upgrade pip wheel
"$VENV/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# ── config ────────────────────────────────────────────────────────────── #
if [ ! -f "$INSTALL_DIR/config.yaml" ]; then
    echo ">> Creating config.yaml from example …"
    cp "$INSTALL_DIR/config.example.yaml" "$INSTALL_DIR/config.yaml"
    echo "   *** Edit $INSTALL_DIR/config.yaml and set your API token ***"
fi

# ── systemd ───────────────────────────────────────────────────────────── #
echo ">> Installing systemd service …"
sudo cp "$INSTALL_DIR/plate-spotter.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable plate-spotter.service

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Edit $INSTALL_DIR/config.yaml  (set api_token, device_id, etc.)"
echo "  2. sudo systemctl start plate-spotter"
echo "  3. sudo journalctl -u plate-spotter -f"
