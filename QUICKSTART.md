# Quick Start Guide - Gaze Estimation API

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
# Install API dependencies
pip install -r requirements-api.txt

# Make sure base gaze estimation dependencies are installed
pip install -r gaze-estimation-testing-main/gaze-estimation/requirements.txt
```

### Step 2: Start the Server

**Windows (Git Bash):**

```bash
bash start_api.sh
```

**Windows (CMD/PowerShell):**

```cmd
start_api.bat
```

**Manual start:**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: Test the API

**Option A - Python test script (recommended):**

```bash
python test_api.py
```

**Option B - Bash test script:**

```bash
bash test_api.sh
```

**Option C - Try it manually in your browser:**

Open http://localhost:8000/docs and use the interactive Swagger UI!

---

## 📝 Usage Examples

### Upload a video via curl:

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@your_video.mp4" \
  -F "model=resnet50"
```

### Upload via Python:

```python
import requests

# Upload video
with open("video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/upload",
        files={"file": f},
        data={"model": "resnet50"}
    )

job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# Check status
import time
while True:
    status = requests.get(f"http://localhost:8000/api/jobs/{job_id}").json()
    if status["status"] in ["completed", "failed"]:
        break
    time.sleep(5)

# Download result
if status["status"] == "completed":
    video = requests.get(f"http://localhost:8000/api/download/{job_id}")
    with open("output.mp4", "wb") as f:
        f.write(video.content)
```

---

## 🔧 Available Models

- `resnet18` - Fast, 43 MB
- `resnet34` - Balanced, 81.6 MB
- `resnet50` - Best accuracy (default), 91.3 MB
- `mobilenetv2` - Lightweight, 9.59 MB
- `mobileone_s0` - Ultra-fast, 4.8 MB

---

## 📊 API Endpoints

| Endpoint                 | Method | Description                   |
| ------------------------ | ------ | ----------------------------- |
| `/api/upload`            | POST   | Upload video for processing   |
| `/api/jobs/{job_id}`     | GET    | Check job status              |
| `/api/jobs/all`          | GET    | Get all jobs                  |
| `/api/download/{job_id}` | GET    | Download processed video      |
| `/api/jobs/{job_id}`     | DELETE | Delete job and files          |
| `/api/health`            | GET    | Check API health              |
| `/docs`                  | GET    | Interactive API documentation |

---

## 🎯 What You Get

The API processes your video and returns it with:

- ✅ Green bounding boxes around detected faces
- ✅ Red arrows showing gaze direction
- ✅ Frame-by-frame gaze estimation
- ✅ Same video format as input

---

## ⚙️ Configuration

Edit `api/config.py` to change:

- Max file size (default: 1GB)
- File retention time (default: 24 hours)
- Default model (default: resnet50)
- Allowed video formats

---

## 🆘 Troubleshooting

**Server won't start?**

- Make sure port 8000 is not in use
- Check if dependencies are installed: `pip list | grep fastapi`

**Model weights not found?**

- Ensure weights exist: `ls gaze-estimation-testing-main/gaze-estimation/weights/resnet50.pt`

**Processing fails?**

- Check API logs for errors
- Verify video format is supported (.mp4, .mov, .avi)
- Ensure enough disk space in `storage/` directory

---

## 📚 Full Documentation

See [API_README.md](API_README.md) for complete documentation.

---

## 🎉 That's It!

You now have a fully functional gaze estimation API running locally. Visit http://localhost:8000/docs to explore the interactive API documentation!
