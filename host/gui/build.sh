#!/bin/bash
# Build script for AgroStake GUI Application (macOS/Linux)
# Usage: bash build.sh

echo ""
echo "========================================"
echo "  AgroStake GUI Builder"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

echo "[1/4] Checking dependencies..."
if ! python3 -c "import PyQt6" 2>/dev/null; then
    echo "[*] Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
else
    echo "[OK] All dependencies are installed"
fi

echo ""
echo "[2/4] Checking PyInstaller..."
if ! pip show pyinstaller > /dev/null; then
    echo "[*] Installing PyInstaller..."
    pip install pyinstaller
fi

echo ""
echo "[3/4] Building executable..."
echo ""
pyinstaller agrostake_gui.spec

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo ""
echo "[4/4] Build complete!"
echo ""
echo "Output: dist/AgroStake/AgroStake"
echo ""
