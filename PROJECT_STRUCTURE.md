# Gaze Estimation API - Project Structure

## Essential Files (Required for API)

```
backend_a_space/
├── api/                              # FastAPI application
│   ├── __init__.py                   # Package initialization
│   ├── config.py                     # Configuration settings
│   ├── gaze_service.py              # Gaze estimation service
│   ├── job_manager.py               # Job tracking and persistence
│   ├── main.py                      # FastAPI endpoints
│   └── models.py                    # Pydantic data models
│
├── gaze-estimation-testing-main/
│   ├── input/                      # Sample input videos (for testing)
│   │   ├── in_train.mp4
│   │   ├── delivery.mp4
│   │   └── speech.mp4
│   └── gaze-estimation/             # Core gaze estimation library
│       ├── models/                  # Neural network architectures
│       │   ├── __init__.py
│       │   ├── mobilenet.py
│       │   ├── mobileone.py
│       │   └── resnet.py
│       ├── utils/                   # Helper functions
│       │   └── helpers.py
│       ├── weights/                 # Model weights (91MB+)
│       │   ├── resnet50.pt         # Primary model
│       │   ├── resnet18_gaze.onnx
│       │   └── mobileone_s0_gaze.onnx
│       ├── config.py               # Dataset configurations
│       └── inference.py            # Core inference logic
│
├── storage/                         # Active storage (auto-managed)
│   ├── uploads/                    # Uploaded videos (temporary)
│   ├── outputs/                    # Processed videos
│   └── jobs.json                   # Job state persistence
│
├── venv/                           # Python virtual environment
│
├── API_README.md                   # Complete API documentation
├── QUICKSTART.md                   # Quick start guide
├── requirements-api.txt            # Python dependencies
├── start_api.sh                    # Startup script (Linux/Mac/Git Bash)
└── start_api.bat                   # Startup script (Windows)
```

## Non-Essential Files (Moved to other_stuff/)

```
other_stuff/
├── test_scripts/                   # API testing scripts
│   ├── test_api.py
│   ├── test_api.sh
│   └── test_output.mp4
│
├── sample_videos/                  # Old output/result videos
│   ├── output/
│   └── results/
│
├── documentation/                  # Original project docs
│   ├── PROJECT_SUMMARY.md
│   └── README.md
│
├── old_scripts/                    # Original CLI scripts
│   ├── run_gaze_estimation.py
│   ├── setup.py
│   └── requirements.txt
│
└── README.md                       # Index of other_stuff contents
```

## Notes

- **Input Videos:** The `gaze-estimation-testing-main/input/` folder contains sample videos used by test scripts and is kept in place for convenience.
- **Storage:** The `storage/` folder is auto-managed by the API and cleaned up after 24 hours.
- **Other Stuff:** Non-essential files are organized in `other_stuff/` for reference.

## Key Features

### API Endpoints
- `POST /api/upload` - Upload video for processing
- `GET /api/jobs/{job_id}` - Check job status
- `GET /api/download/{job_id}` - Download processed video
- `DELETE /api/jobs/{job_id}` - Delete job
- `GET /api/health` - Health check
- `GET /docs` - Interactive API documentation

### Storage Management
- Automatic cleanup of files older than 24 hours
- Persistent job tracking across server restarts
- Single-video-at-a-time processing queue

### Supported Models
- ResNet-50 (default) - 91.3 MB, 11.34° MAE
- ResNet-18 - 43 MB, 12.84° MAE
- ResNet-34 - 81.6 MB, 11.33° MAE
- MobileNet V2 - 9.59 MB, 13.07° MAE
- MobileOne S0 - 4.8 MB, 12.58° MAE

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements-api.txt
   ```

2. **Start the server:**
   ```bash
   bash start_api.sh
   # or
   start_api.bat
   ```

3. **Access the API:**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

## Testing

Use test scripts from `other_stuff/test_scripts/`:
```bash
python other_stuff/test_scripts/test_api.py
```

## Documentation

- **API_README.md** - Complete API documentation
- **QUICKSTART.md** - Quick start guide
- **other_stuff/documentation/** - Original project documentation

