#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Audio2Face Web UI Startup Script
# This script starts both the backend API server and opens the frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEBUI_DIR="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/venv"

echo "================================================"
echo "    Audio2Face Web UI"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Please create it first with: python -m venv venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if required packages are installed
echo "Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing FastAPI and dependencies..."
    pip install fastapi uvicorn python-multipart pydub
fi

# Create required directories
mkdir -p "$WEBUI_DIR/uploads"
mkdir -p "$WEBUI_DIR/results"

# Start the backend server
echo ""
echo "Starting backend server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
echo ""

cd "$WEBUI_DIR/backend"

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Warning: Port 8000 is already in use."
    echo "Attempting to kill existing process..."
    kill $(lsof -Pi :8000 -sTCP:LISTEN -t) 2>/dev/null || true
    sleep 1
fi

# Start uvicorn in background
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
sleep 3

# Check if server started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Error: Failed to start backend server"
    exit 1
fi

echo ""
echo "================================================"
echo "    Server Started Successfully!"
echo "================================================"
echo ""
echo "Backend API: http://localhost:8000"
echo "API Docs:    http://localhost:8000/docs"
echo ""
echo "To open the frontend, open this file in a browser:"
echo "  $WEBUI_DIR/frontend/index.html"
echo ""
echo "Or start a simple HTTP server for the frontend:"
echo "  cd $WEBUI_DIR/frontend && python -m http.server 3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Open frontend in browser (if available)
if command -v xdg-open &> /dev/null; then
    # Try to start a simple HTTP server for frontend and open browser
    cd "$WEBUI_DIR/frontend"
    python -m http.server 3000 &
    FRONTEND_PID=$!
    sleep 1
    xdg-open "http://localhost:3000" 2>/dev/null || true
fi

# Trap Ctrl+C to cleanup
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    deactivate 2>/dev/null || true
    echo "Done."
    exit 0
}

trap cleanup INT TERM

# Wait for backend process
wait $BACKEND_PID
