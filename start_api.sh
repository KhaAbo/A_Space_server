#!/bin/bash
# Startup script for Gaze Estimation API

echo "Starting Gaze Estimation API..."
echo ""

# Navigate to project root
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Warning: No virtual environment found. It's recommended to create one:"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate  # or venv\\Scripts\\activate on Windows"
    echo ""
fi

# Check if requirements are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing API dependencies..."
    pip install -r requirements-api.txt
    echo ""
fi

# Start the API server
echo "Starting server at http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
echo ""

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

