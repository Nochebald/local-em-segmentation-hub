#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PID_FILE="$DIR/app.pid"

echo "Stopping Local EM Segmentation & 3D Reconstruction Hub..."

if [ ! -f "$PID_FILE" ]; then
    echo "No running application found."
    exit 0
fi

APP_PID=$(cat "$PID_FILE")

if kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID"
    echo "Application stopped."
else
    echo "Process $APP_PID is no longer running."
fi

rm -f "$PID_FILE"
