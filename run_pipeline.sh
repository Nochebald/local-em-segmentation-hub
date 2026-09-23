#!/bin/bash

# Find the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Starting Local EM Segmentation & 3D Reconstruction Hub..."

# Check that the virtual environment exists
if [ ! -f "$DIR/env/bin/activate" ]; then
    echo ""
    echo "Error: Python virtual environment not found."
    echo ""
    echo "Create it first with:"
    echo "  python3 -m venv env"
    echo "  source env/bin/activate"
    echo "  pip install -r requirements.txt"
    echo ""
    exit 1
fi

echo "Activating virtual environment: $DIR/env"

source "$DIR/env/bin/activate"

echo "Launching application..."
python app.py
