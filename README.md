# A Space Server - Gaze Estimation API

FastAPI-based service for video gaze estimation using deep learning models. Processes videos frame-by-frame to detect faces and predict gaze directions, outputting annotated videos with gaze direction arrows.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [API Documentation](#api-documentation)
- [Technical Details](#technical-details)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Docker Setup (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/KhaAbo/A_Space_server
cd A_Space_server

# 2. Pull model weights with Git LFS
git lfs install
git lfs pull

# 3. Set up Discord webhook (optional)
cp api/env.example api/.env
# Edit api/.env and add your Discord webhook URL

# 4. Start the API
docker-compose up --build
```

**Common Commands:**

```bash
docker-compose up --build      # Start API
docker-compose up -d           # Run in background
docker-compose down            # Stop API
docker-compose logs -f         # View logs
```

### Local Setup

**⚠️ Requires Python 3.10-3.13.2** (Python 3.13.5+ has compatibility issues)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-api.txt

# 3. Start server
bash start_api.sh  # Windows: start_api.bat

# Or manually:
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access at: **http://localhost:8000** | Docs: **http://localhost:8000/docs**

---

## Features

- 📹 Upload videos for gaze estimation processing
- 🎯 Multiple model options (ResNet18/34/50, MobileNet, MobileOne)
- 📊 Async job processing with frame-by-frame progress tracking
- 💾 Download processed videos with gaze annotations
- 🧹 Automatic cleanup (24-hour retention)
- 🔒 Single-video-at-a-time queue management

### Supported Models

| Model               | Size    | Accuracy (MAE) | Speed      |
| ------------------- | ------- | -------------- | ---------- |
| ResNet-50 (default) | 91.3 MB | 11.34°         | Moderate   |
| ResNet-18           | 43 MB   | 12.84°         | Fast       |
| MobileOne S0        | 4.8 MB  | 12.58°         | Ultra-Fast |

---

## API Documentation

See **[API_README.md](API_README.md)** for complete API reference.

**Endpoints:**

- `POST /api/upload` - Upload video for processing
- `GET /api/jobs/{job_id}` - Check job status with progress tracking
- `GET /api/jobs/all` - Get all jobs
- `GET /api/download/{job_id}` - Download processed video
- `DELETE /api/jobs/{job_id}` - Delete job and files
- `GET /api/health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)

**Quick Example:**

```python
import requests
import time

API_URL = "http://localhost:8000"

# Upload video
with open("video.mp4", "rb") as f:
    job_id = requests.post(f"{API_URL}/api/upload",
                          files={"file": f},
                          data={"model": "resnet50"}).json()["job_id"]

# Monitor progress
while True:
    job = requests.get(f"{API_URL}/api/jobs/{job_id}").json()
    if job['status'] == 'processing' and job['progress_percentage']:
        print(f"Progress: {job['progress_percentage']:.1f}%")
    if job['status'] in ['completed', 'failed']:
        break
    time.sleep(2)

# Download result
if job['status'] == 'completed':
    with open("processed.mp4", "wb") as f:
        f.write(requests.get(f"{API_URL}/api/download/{job_id}").content)
```

---

## Technical Details

### Processing Pipeline

1. **Face Detection:** RetinaFace (MobileNet V2) detects faces in each frame
2. **Gaze Estimation:** For each face:
   - Crop and preprocess (resize to 448×448, normalize)
   - Run through gaze model (ResNet-50/MobileOne/etc.)
   - Model outputs pitch/yaw angles via binned classification (90 bins, 4° width)
   - Convert to gaze direction vector
3. **Visualization:** Draw bounding boxes and gaze direction arrows
4. **Output:** Save annotated video to `storage/outputs/{job_id}/processed.mp4`

### Key Features

- **Binned Classification:** Gaze360 dataset (90 bins, ±180° range)
- **GPU Acceleration:** Auto-detects CUDA, falls back to CPU
- **Singleton Models:** Models loaded once and reused for efficiency
- **Progress Tracking:** Updates every 10 frames (configurable)

### Performance (ResNet-50)

- **Accuracy:** 11.34° average angular error
- **Speed:** CPU ~2-5 FPS | GPU ~30-60 FPS
- **Memory:** ~500-800 MB GPU memory

---

## Configuration

Edit `api/config.py`:

- `MAX_FILE_SIZE = 1024 * 1024 * 1024` (1GB)
- `ALLOWED_FORMATS = [".mp4", ".mov", ".avi"]`
- `DEFAULT_MODEL = "resnet50"`
- `FILE_RETENTION_HOURS = 24`
- `CLEANUP_INTERVAL_HOURS = 1`
- Progress update frequency: `api/gaze_service.py` line 118 (default: every 10 frames)

---

## Troubleshooting

**Port 8000 in use:**

```bash
# Change port in docker-compose.yml or startup command
uvicorn api.main:app --reload --port 8001
```

**Model weights not found:**

```bash
git lfs install
git lfs pull
# Verify: ls gaze-estimation-testing-main/gaze-estimation/weights/resnet50.pt
```

**Import errors:**

```bash
pip install -r requirements-api.txt
pip install -r gaze-estimation-testing-main/gaze-estimation/requirements.txt
```

**GPU not detected:**

```bash
python -c "import torch; print(torch.cuda.is_available())"
# For Docker: Install NVIDIA Container Toolkit
```

**Docker issues:**

- Container restarting: Check logs with `docker-compose logs -f`
- Missing weights: Run `git lfs pull`
- Port conflict: Change port in `docker-compose.yml`

**Processing fails:**

- Check API logs
- Verify video format (.mp4, .mov, .avi)
- Ensure sufficient disk space
- Verify weights are downloaded (not LFS pointers)

---

## Credits

Based on **MobileGaze** (Valikhujaev, 2024):

- GitHub: https://github.com/yakhyo/gaze-estimation
- DOI: 10.5281/zenodo.14257640

```bibtex
@misc{valikhujaev2024mobilegaze,
  author       = {Valikhujaev, Y.},
  title        = {MobileGaze: Pre-trained mobile nets for Gaze-Estimation},
  year         = {2024},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14257640},
  url          = {https://doi.org/10.5281/zenodo.14257640}
}
```
