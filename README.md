# A Space Server - Gaze Estimation API

FastAPI-based service for video gaze estimation using deep learning models.

## Quick Start

1. Install dependencies:

   pip install -r requirements-api.txt 2. Start the server:
   bash start_api.sh

   # or on Windows: start_api.bat

   3. Access the API:

   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

## Documentation

- [API Documentation](API_README.md) - Complete API reference
- [Quick Start Guide](QUICKSTART.md) - Getting started guide
- [Project Structure](PROJECT_STRUCTURE.md) - File organization

## Features

- 📹 Upload videos for gaze estimation processing
- 🎯 Multiple model options (ResNet18/34/50, MobileNet, MobileOne)
- 📊 Async job processing with status tracking
- 💾 Download processed videos with gaze annotations
- 🧹 Automatic cleanup of old files (24-hour retention)

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)
- Model weights in `gaze-estimation-testing-main/gaze-estimation/weights/`

## License

[Add your license here]
