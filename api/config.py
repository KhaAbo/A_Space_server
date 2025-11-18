import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env file in the api directory first, then fall back to parent directory
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Base paths
BASE_DIR = Path(__file__).parent.parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
OUTPUTS_DIR = STORAGE_DIR / "outputs"
JOBS_FILE = STORAGE_DIR / "jobs.json"

# Gaze estimation paths
GAZE_ESTIMATION_DIR = BASE_DIR / "gaze-estimation-testing-main" / "gaze-estimation"
WEIGHTS_DIR = GAZE_ESTIMATION_DIR / "weights"

# File upload settings
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB in bytes
ALLOWED_FORMATS = [".mp4", ".mov", ".avi"]

# Model settings
DEFAULT_MODEL = "resnet50"
SUPPORTED_MODELS = ["resnet18", "resnet34", "resnet50", "mobilenetv2", "mobileone_s0"]

# Dataset configuration (fixed to gaze360)
DATASET = "gaze360"
BINS = 90
BINWIDTH = 4
ANGLE = 180

# Cleanup settings
CLEANUP_INTERVAL_HOURS = 1
FILE_RETENTION_HOURS = 24

# Discord webhook settings
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", None)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)