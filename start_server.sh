#!/bin/bash
set -e

echo "Starting gaze estimation API..."
echo "Working directory: $(pwd)"
echo "Python path: $PYTHONPATH"
echo "Python version: $(python --version)"

# Test imports
echo "Testing imports..."
python -c "import sys; sys.path.insert(0, '/app'); import api.main" || {
    echo "Import failed!"
    exit 1
}

echo "Starting uvicorn server..."
exec python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

