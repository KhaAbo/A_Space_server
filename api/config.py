import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in root directory
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
STORAGE_DIR = Path(f"{BASE_DIR}/storage")
UPLOADS_DIR = STORAGE_DIR / "uploads"
OUTPUTS_DIR = STORAGE_DIR / "outputs"
JOBS_FILE = STORAGE_DIR / "jobs.json"

# Robin's model
MOBILEGAZE_DIR = f"{BASE_DIR}/mobilegaze"
MOBILEGAZE_CONFIG = f"{MOBILEGAZE_DIR}/config/gaze_config.yml"

# Björn's model
MOGCNN_DIR = f"{BASE_DIR}/mog-eyecontact"
MOGCNN_CONFIG = f"{MOGCNN_DIR}/config/config.yml"

# File upload settings
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB in bytes
ALLOWED_FORMATS = [".mp4", ".mov", ".avi"]

# Model settings
DEFAULT_MODEL = "resnet50"
SUPPORTED_MODELS = ["resnet18", "resnet50", "mobileone_s0"]

# Cleanup settings
CLEANUP_INTERVAL_HOURS = 1
FILE_RETENTION_HOURS = 24

# Discord webhook settings from .env file
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Create directories if they don't exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
DEFAULT_MODEL = "resnet50"
SUPPORTED_MODELS = ["resnet50", "eye_contact"]