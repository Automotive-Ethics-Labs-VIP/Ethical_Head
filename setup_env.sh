#!/bin/bash
# AEL Team A - Environment Setup (Python 3.10)

set -e

echo "Creating virtual environment..."
python3.10 -m venv venv

echo "Activating..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Done! Activate with: source venv/bin/activate"
