# A Space Server - Gaze Estimation API

FastAPI service for video gaze estimation. Detects faces and predicts gaze directions, outputting annotated videos.

## Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/KhaAbo/A_Space_server
cd A_Space_server

# Pull model weights
git lfs install && git lfs pull
# If LFS issues, download manually: https://drive.google.com/file/d/1iXMWdS9HwRDW7OLKN-LGr7N1ghgfT59k/view

# Start API
docker-compose up --build
```

### Local Setup

**Requires Python 3.10-3.13.2**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-docker.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Access:** http://localhost:8000 | **Docs:** http://localhost:8000/docs

## API Endpoints

| Endpoint                          | Description                     |
| --------------------------------- | ------------------------------- |
| `POST /api/upload`                | Upload video (returns `job_id`) |
| `GET /api/jobs/{job_id}`          | Check status & progress         |
| `GET /api/download/{job_id}`      | Download processed video        |
| `GET /api/download-data/{job_id}` | Download gaze data JSON         |
| `DELETE /api/jobs/{job_id}`       | Delete job and files            |
| `GET /api/health`                 | Health check                    |

See [API_README.md](API_README.md) for full documentation.

### Quick Example

```python
import requests, time

API = "http://localhost:8000"

# Upload
job_id = requests.post(f"{API}/api/upload", files={"file": open("video.mp4", "rb")}).json()["job_id"]

# Wait for completion
while (job := requests.get(f"{API}/api/jobs/{job_id}").json())["status"] not in ["completed", "failed"]:
    print(f"Progress: {job.get('progress_percentage', 0):.1f}%")
    time.sleep(2)

# Download
if job["status"] == "completed":
    open("output.mp4", "wb").write(requests.get(f"{API}/api/download/{job_id}").content)
```

## Configuration

Edit `api/config.py`:

| Setting                | Default          | Description            |
| ---------------------- | ---------------- | ---------------------- |
| `MAX_FILE_SIZE`        | 1GB              | Max upload size        |
| `ALLOWED_FORMATS`      | .mp4, .mov, .avi | Accepted video formats |
| `DEFAULT_MODEL`        | resnet50         | Gaze estimation model  |
| `FILE_RETENTION_HOURS` | 24               | Auto-cleanup after     |

## Troubleshooting

| Issue             | Solution                                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| Port in use       | `uvicorn api.main:app --port 8001`                                                                            |
| Weights not found | `git lfs pull` or [download manually](https://drive.google.com/file/d/1iXMWdS9HwRDW7OLKN-LGr7N1ghgfT59k/view) |
| Import errors     | `pip install -r requirements-docker.txt`                                                                      |
| GPU not detected  | `python -c "import torch; print(torch.cuda.is_available())"`                                                  |

## Credits

Based on [MobileGaze](https://github.com/yakhyo/gaze-estimation) (Valikhujaev, 2024) - DOI: 10.5281/zenodo.14257640
