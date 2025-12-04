# Gaze Estimation API

A FastAPI-based service for video gaze estimation using deep learning models.

## Features

- 📹 Upload videos for gaze estimation processing
- 🎯 Process videos with multiple model options (ResNet18/34/50, MobileNet, MobileOne)
- 📊 Track job status asynchronously with real-time progress tracking
- 📈 Frame-by-frame progress monitoring (total frames, processed frames, percentage)
- 💾 Download processed videos with annotated gaze directions
- 🧹 Automatic cleanup of old files (24-hour retention)
- 🔒 Single-video-at-a-time processing to prevent resource overload

## Quick Start

### 1. Install Dependencies

First, install the base gaze estimation dependencies:

```bash
pip install -r mobilegaze/requirements.txt

```

Then install the API dependencies:

```bash
pip install -r requirements-api.txt
```

### 2. Ensure Model Weights Exist

Make sure you have model weights downloaded in:

```
mobilegaze/weights/resnet50.pt

```

### 3. Start the API Server

**On Windows (Git Bash):**

```bash
bash start_api.sh
```

**On Windows (CMD/PowerShell):**

```cmd
start_api.bat
```

**Or manually:**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

- **API**: http://localhost:8000
- **Swagger UI Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### `POST /api/upload`

Upload a video file for processing.

**Request:**

- `file` (form-data): Video file (.mp4, .mov, .avi)
- `model` (optional): Model name (default: resnet50)
  - Options: `resnet18`, `resnet34`, `resnet50`, `mobilenetv2`, `mobileone_s0`

**Response:**

```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "message": "Video uploaded successfully. Processing started."
}
```

**Example (curl):**

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@video.mp4" \
  -F "model=resnet50"
```

**Example (Python):**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/upload",
    files={"file": open("video.mp4", "rb")},
    data={"model": "resnet50"}
)
job_id = response.json()["job_id"]
```

---

### `GET /api/jobs/{job_id}`

Check the status of a processing job with real-time progress tracking.

**Response:**

```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "filename": "video.mp4",
  "model": "resnet50",
  "created_at": "2025-11-17T10:30:00",
  "started_at": "2025-11-17T10:30:05",
  "completed_at": null,
  "error": null,
  "total_frames": 1500,
  "processed_frames": 750,
  "progress_percentage": 50.0
}
```

**Status values:**

- `pending`: Waiting to be processed
- `processing`: Currently being processed
- `completed`: Successfully completed
- `failed`: Processing failed (check `error` field)

**Progress tracking fields:**

- `total_frames` (int|null): Total number of frames in the video (populated once processing starts, `null` while job is in `pending` status)
- `processed_frames` (int): Number of frames processed so far (starts at `0`, updates every 10 frames during processing, final value equals `total_frames` when completed)
- `progress_percentage` (float|null): Completion percentage (0-100), calculated as `(processed_frames / total_frames) * 100` (rounded to 2 decimal places, `null` until processing begins)

> **Note:** Progress updates occur every 10 frames to balance real-time feedback with performance. This frequency can be adjusted in `api/gaze_service.py` line 118.

**Example:**

```bash
curl http://localhost:8000/api/jobs/{job_id}
```

---

### `GET /api/jobs/all`

Get all jobs.

**Response:** Array of job objects

[
{
"job_id": "123e4567-e89b-12d3-a456-426614174000",
"status": "processing",
"filename": "video.mp4",
"model": "resnet50",
"created_at": "2025-11-17T10:30:00",
"started_at": "2025-11-17T10:30:05",
"completed_at": null,
"error": null,
"total_frames": 1500,
"processed_frames": 750,
"progress_percentage": 50.0
},
{
"job_id": "456e7890-e89b-12d3-a456-426614174001",
"status": "completed",
"filename": "video2.mp4",
"model": "resnet50",
"created_at": "2025-11-17T09:00:00",
"started_at": "2025-11-17T09:00:05",
"completed_at": "2025-11-17T09:05:30",
"error": null,
"total_frames": 2400,
"processed_frames": 2400,
"progress_percentage": 100.0
}
]**Example:**

## curl http://localhost:8000/api/jobs/all

### `GET /api/download/{job_id}`

Download the processed video (only available when status is `completed`).

**Response:** MP4 video file

**Example:**

```bash
curl http://localhost:8000/api/download/{job_id} -o processed_video.mp4
```

---

### `DELETE /api/jobs/{job_id}`

Delete a job and its associated files.

**Response:** 204 No Content

**Example:**

```bash
curl -X DELETE http://localhost:8000/api/jobs/{job_id}
```

---

### `GET /api/health`

Check API health and system status.

**Response:**

```json
{
  "status": "healthy",
  "gpu_available": true,
  "model_loaded": false,
  "storage_path": "/path/to/storage"
}
```

---

## Complete Workflow Example

### Using Bash Script

Run the included test script:

```bash
bash test_api.sh
```

This will:

1. Check API health
2. Upload a test video
3. Poll for completion
4. Download the result

### Using Python

```python
import requests
import time

API_URL = "http://localhost:8000"

# 1. Upload video
with open("video.mp4", "rb") as f:
    response = requests.post(
        f"{API_URL}/api/upload",
        files={"file": f},
        data={"model": "resnet50"}
    )
job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# 2. Poll for status with progress monitoring
while True:
    response = requests.get(f"{API_URL}/api/jobs/{job_id}")
    job = response.json()

    # Display progress information
    status = job['status']
    if status == 'processing' and job['progress_percentage']:
        print(f"Status: {status} - {job['progress_percentage']:.1f}% "
              f"({job['processed_frames']}/{job['total_frames']} frames)")
    else:
        print(f"Status: {status}")

    if job["status"] in ["completed", "failed"]:
        break

    time.sleep(2)  # Check every 2 seconds for progress updates

# 3. Download result
if job["status"] == "completed":
    response = requests.get(f"{API_URL}/api/download/{job_id}")
    with open("processed.mp4", "wb") as f:
        f.write(response.content)
    print("Download complete!")
else:
    print(f"Processing failed: {job['error']}")
```

## Configuration

Edit `api/config.py` to customize:

- **File size limit**: `MAX_FILE_SIZE = 1024 * 1024 * 1024` (1GB)
- **Allowed formats**: `ALLOWED_FORMATS = [".mp4", ".mov", ".avi"]`
- **Default model**: `DEFAULT_MODEL = "resnet50"`
- **File retention**: `FILE_RETENTION_HOURS = 24`
- **Cleanup interval**: `CLEANUP_INTERVAL_HOURS = 1`

## File Structure

```
backend_a_space/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models.py            # Pydantic data models
│   ├── job_manager.py       # Job tracking and persistence
│   └── gaze_service.py      # Gaze estimation service
├── storage/
│   ├── uploads/             # Uploaded videos (temporary)
│   ├── outputs/             # Processed videos
│   └── jobs.json            # Job state persistence
├── requirements-api.txt     # API dependencies
├── start_api.sh             # Startup script (Linux/Mac/Git Bash)
├── start_api.bat            # Startup script (Windows)
└── test_api.sh              # API testing script
```

## Technical Details

### Processing Pipeline

1. Video uploaded → saved to `storage/uploads/{job_id}/`
2. Job created with status `pending`
3. Background task starts processing:
   - Status changes to `processing`
   - Total frames counted and stored
   - Face detection on each frame (RetinaFace)
   - Gaze estimation for each detected face
   - Draw bounding boxes and gaze arrows
   - Progress updated every 10 frames
   - Save to `storage/outputs/{job_id}/processed.mp4`
4. Status changes to `completed` or `failed`
5. Progress set to 100% on completion

### Single-Video Processing

The API uses an `asyncio.Lock()` to ensure only one video is processed at a time, preventing GPU/CPU resource conflicts. Additional uploads will queue in `pending` status.

### Automatic Cleanup

- On startup: Removes jobs older than 24 hours
- Background task: Runs cleanup every hour
- Deletes both job records and associated files

### Model Loading

Models are loaded once on first use (singleton pattern) and kept in memory for subsequent requests. This improves processing speed for multiple videos.

## Limits (Internal Use Configuration)

- **Max file size**: 1GB
- **Concurrent processing**: 1 video at a time
- **File retention**: 24 hours
- **Supported formats**: MP4, MOV, AVI
- **Dataset**: Gaze360 (fixed)
- **Authentication**: None (internal use only)

## Troubleshooting

### Port already in use

```bash
# Change port in startup command
uvicorn api.main:app --reload --port 8001
```

### Model weights not found

```bash
# Ensure weights exist
ls mobilegaze/weights/resnet50.pt

```

### Import errors

```bash
# Reinstall dependencies
pip install -r requirements-api.txt
pip install -r mobilegaze/requirements.txt

```

### GPU not detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Development

### Interactive API Documentation

FastAPI automatically generates interactive documentation:

- **Swagger UI**: http://localhost:8000/docs

  - Try out endpoints directly in the browser
  - See request/response schemas
  - Download OpenAPI specification

- **ReDoc**: http://localhost:8000/redoc
  - Alternative documentation view
  - Better for reading/reference

### Adding New Models

1. Add model name to `SUPPORTED_MODELS` in `api/config.py`
2. Ensure weights exist in `mobilegaze/weights/{model}.pt`

3. Model will be automatically available in upload endpoint

### Logging

The API logs to stdout. View logs to monitor processing:

```bash
# Logs show:
# - Model loading status
# - Frame processing progress
# - Job status changes
# - Cleanup operations
```

## License

This API wraps the MobileGaze gaze estimation library. See the original repository for licensing information.

## Credits

Based on **MobileGaze** (Valikhujaev, 2024):

- GitHub: https://github.com/yakhyo/gaze-estimation
- DOI: 10.5281/zenodo.14257640
