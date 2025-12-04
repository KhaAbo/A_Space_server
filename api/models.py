from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    model: str # Legacy field
    face_model: Optional[str] = None
    gaze_model: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    total_frames: Optional[int] = None
    processed_frames: int = 0
    progress_percentage: Optional[float] = None


class UploadResponse(BaseModel):
    job_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    model_loaded: bool
    storage_path: str