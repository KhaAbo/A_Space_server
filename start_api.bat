@echo off
REM Startup script for Gaze Estimation API (Windows)

echo Starting Gaze Estimation API...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Warning: No virtual environment found. It's recommended to create one:
    echo   python -m venv venv
    echo   venv\Scripts\activate
    echo.
)

REM Check if FastAPI is installed
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo Installing API dependencies...
    pip install -r requirements-docker.txt
    echo.
)

REM Start the API server
echo Starting server at http://localhost:8000
echo API docs available at http://localhost:8000/docs
echo.

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

