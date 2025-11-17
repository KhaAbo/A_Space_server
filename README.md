# A Space Server - Gaze Estimation API

FastAPI-based service for video gaze estimation using deep learning models.

## 🐳 Quick Start with Docker (Recommended)

```bash
# 1. Clone and navigate to the repository
git clone https://github.com/KhaAbo/A_Space_server
cd A_Space_server

# 2. Make sure Git LFS is installed and pull model weights
git lfs install
git lfs pull

# 3. Start the API with Docker
docker-compose up --build

# 4. Access the API at http://localhost:8000
```

That's it! The API is now running in a consistent environment. 🎉

## 💻 Alternative: Local Setup

**⚠️ Note:** For best compatibility, use **Python 3.10-3.13.2**. Python 3.13.5+ may have compatibility issues.

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-api.txt

# 3. Start the server
bash start_api.sh  # Windows: start_api.bat

# 4. Access the API
```

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

### For Docker (Recommended):
- Docker and Docker Compose
- Git LFS (for model weights)

### For Local Setup:
- Python 3.10-3.13.2 (⚠️ avoid Python 3.13.5+)
- CUDA-capable GPU (optional, but recommended)
- Model weights in `gaze-estimation-testing-main/gaze-estimation/weights/`

## 🐋 Docker Commands

```bash
# Start the API
docker-compose up

# Start in detached mode (background)
docker-compose up -d

# Stop the API
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after code changes
docker-compose up --build

# Remove everything (including volumes)
docker-compose down -v
```

## License

[Add your license here]
