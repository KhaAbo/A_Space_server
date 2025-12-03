import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional
import subprocess


from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Query,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import torch

from api.config import (
    MAX_FILE_SIZE,
    ALLOWED_FORMATS,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    UPLOADS_DIR,
    OUTPUTS_DIR,
    BINS,
    BINWIDTH,
    ANGLE,
    WEIGHTS_DIR,
    ensure_directories,
    CLEANUP_INTERVAL_HOURS,
)
from api.models import JobStatus, JobInfo, UploadResponse, HealthResponse
from api.job_manager import JobManager
from api.gaze_service import gaze_service
from api.discord_webhook import send_job_notification


# Initialize FastAPI app
app = FastAPI(
    title="Gaze Estimation API",
    description="API for video-based gaze estimation using deep learning",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
job_manager: Optional[JobManager] = None
processing_lock = asyncio.Lock()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global job_manager

    # Ensure directories exist
    ensure_directories()

    # Initialize job manager
    job_manager = JobManager()

    # Run cleanup of old files
    cleaned = job_manager.cleanup_old_jobs()
    print(f"Startup cleanup: removed {cleaned} old jobs")

    # Start background cleanup task
    asyncio.create_task(periodic_cleanup())

    print("API started successfully")


async def periodic_cleanup():
    """Periodically clean up old jobs."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_HOURS * 3600)
        cleaned = job_manager.cleanup_old_jobs()
        if cleaned > 0:
            print(f"Periodic cleanup: removed {cleaned} old jobs")


async def process_video_background(job_id: str, input_path: Path, model: str):
    """Background task to process video."""
    async with processing_lock:  # Ensure only one video processes at a time
        try:
            # Update status to processing
            job_manager.update_job_status(job_id, JobStatus.PROCESSING)

            # Setup paths
            output_dir = OUTPUTS_DIR / job_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "processed.mp4"

            # Get weight path
            weight_path = WEIGHTS_DIR / f"{model}.pt"
            if not weight_path.exists():
                raise FileNotFoundError(f"Model weights not found: {weight_path}")

            # Create progress callback function
            def progress_callback(total_frames: int, processed_frames: int):
                """Callback to update job progress."""
                job_manager.update_progress(job_id, total_frames, processed_frames)

            # Process video (blocking call in executor to not block event loop)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                gaze_service.process_video,
                str(input_path),
                str(output_path),
                model,
                BINS,
                BINWIDTH,
                ANGLE,
                str(weight_path),
                progress_callback,
            )

            # Add faststart atom for streaming compatibility
            await add_faststart_to_video(output_path)

            # Update status to completed
            job_manager.update_job_status(job_id, JobStatus.COMPLETED)

            # Send Discord webhook notification
            job = job_manager.get_job(job_id)
            if job:
                asyncio.create_task(send_job_notification(job))

        except Exception as e:
            error_msg = str(e)
            print(f"Error processing job {job_id}: {error_msg}")
            job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_msg)

            # Send Discord webhook notification
            job = job_manager.get_job(job_id)
            if job:
                asyncio.create_task(send_job_notification(job))


@app.post("/api/upload", response_model=UploadResponse)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model: str = DEFAULT_MODEL,
):
    """
    Upload a video file for gaze estimation processing.

    - **file**: Video file (.mp4, .mov, .avi)
    - **model**: Model to use (default: resnet50)
    """
    # Validate model
    if model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Supported models: {', '.join(SUPPORTED_MODELS)}",
        )

    # Validate file format
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {', '.join(ALLOWED_FORMATS)}",
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Create upload directory
    upload_dir = UPLOADS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    file_path = upload_dir / f"original{file_ext}"

    try:
        # Read and save file (with size check)
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024**3):.1f}GB",
            )

        with open(file_path, "wb") as f:
            f.write(content)

    except Exception as e:
        # Cleanup on error
        if upload_dir.exists():
            import shutil

            shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    # Create job record
    job_manager.create_job(job_id, file.filename, model)

    # Start background processing
    background_tasks.add_task(process_video_background, job_id, file_path, model)

    return UploadResponse(
        job_id=job_id, message="Video uploaded successfully. Processing started."
    )


@app.get("/api/jobs/all", response_model=list[JobInfo])
async def get_all_jobs():
    """
    Get all jobs.

    Returns a list of all jobs with their information including status, timestamps, and errors if failed.
    """
    jobs = job_manager.list_jobs()

    # Parse datetime strings for each job
    job_infos = []
    for job in jobs:
        job_info = JobInfo(
            job_id=job["job_id"],
            status=JobStatus(job["status"]),
            filename=job["filename"],
            model=job["model"],
            created_at=datetime.fromisoformat(job["created_at"]),
            started_at=(
                datetime.fromisoformat(job["started_at"]) if job["started_at"] else None
            ),
            completed_at=(
                datetime.fromisoformat(job["completed_at"])
                if job["completed_at"]
                else None
            ),
            error=job.get("error"),
            total_frames=job.get("total_frames"),
            processed_frames=job.get("processed_frames", 0),
            progress_percentage=job.get("progress_percentage"),
        )
        job_infos.append(job_info)

    return job_infos


@app.get("/api/jobs/{job_id}", response_model=JobInfo)
async def get_job_status(job_id: str):
    """
    Get the status of a processing job.

    Returns job information including status, timestamps, and error if failed.
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Parse datetime strings
    job_info = JobInfo(
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        filename=job["filename"],
        model=job["model"],
        created_at=datetime.fromisoformat(job["created_at"]),
        started_at=(
            datetime.fromisoformat(job["started_at"]) if job["started_at"] else None
        ),
        completed_at=(
            datetime.fromisoformat(job["completed_at"]) if job["completed_at"] else None
        ),
        error=job.get("error"),
        total_frames=job.get("total_frames"),
        processed_frames=job.get("processed_frames", 0),
        progress_percentage=job.get("progress_percentage"),
    )

    return job_info


@app.get("/api/download/{job_id}")
async def download_result(job_id: str):
    """
    Stream or download the processed video file.

    Supports HTTP Range requests for video streaming and seeking.
    Only available when job status is 'completed'.
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}",
        )

    # Check if output file exists
    output_path = OUTPUTS_DIR / job_id / "processed.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")

    return FileResponse(
        path=output_path, media_type="video/mp4", filename=f"gaze_{job['filename']}"
    )


@app.get("/api/download-data/{job_id}")
async def download_gaze_data(job_id: str):
    """
    Download the gaze data JSON file.
    Only available when job status is 'completed'.

    Returns compact JSON with:
    - meta: {fps, total_frames}
    - detections: Only frames where person looked at ROI
    - tracks: Per-person engagement summary
    - summary: Overall video statistics
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job['status']}",
        )

    # Check if output file exists
    json_path = OUTPUTS_DIR / job_id / "gaze_data.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Gaze data not found")

    return FileResponse(
        path=json_path, media_type="application/json", filename=f"gaze_data_{job['filename']}.json"
    )


async def add_faststart_to_video(video_path: Path):
    """
    Re-encode video with H.264 codec and add faststart atom for streaming compatibility.
    This converts OpenCV's mpeg4 codec to libx264 (H.264) which is standard for web video,
    and adds the faststart atom for HTML5 video players.
    """
    temp_path = video_path.with_suffix(".tmp.mp4")

    try:
        # Re-encode with H.264 codec and add faststart atom
        # -c:v libx264: Use H.264 codec for video (web standard)
        # -preset fast: Balance between encoding speed and file size
        # -crf 22: Quality setting (lower = better quality, 18-28 is typical range)
        # -c:a aac: Use AAC codec for audio (if present)
        # -movflags faststart: Add faststart atom for streaming
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-c:a",
            "aac",
            "-movflags",
            "faststart",
            "-y",  # Overwrite output file
            str(temp_path),
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await proc.wait()

        if proc.returncode == 0:
            # Replace original file with re-encoded version
            temp_path.replace(video_path)
            print(f"Re-encoded video with H.264 and added faststart to {video_path}")
        else:
            # If ffmpeg fails, keep original file
            stderr = await proc.stderr.read()
            print(f"Warning: Failed to re-encode video {video_path}: {stderr.decode()}")
            if temp_path.exists():
                temp_path.unlink()
    except Exception as e:
        print(f"Error re-encoding video: {e}")
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/play_original/{job_id}")
async def original_video_stream(job_id: str):
    """
    Stream the original uploaded video file with proper range request support for HTML5 video players.
    Uses FileResponse which automatically handles HTTP range requests correctly.
    Serves the original video from the uploads directory.
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Find the original video file (could be .mp4, .mov, .avi)
    upload_dir = UPLOADS_DIR / job_id
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="Original video not found")
    
    # Look for original video with any allowed extension
    video_path = None
    for ext in ALLOWED_FORMATS:
        potential_path = upload_dir / f"original{ext}"
        if potential_path.exists():
            video_path = potential_path
            break
    
    if not video_path:
        raise HTTPException(status_code=404, detail="Original video not found")
    
    # Determine media type based on extension
    media_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
    }
    media_type = media_types.get(video_path.suffix.lower(), "video/mp4")
    
    return FileResponse(
        path=str(video_path),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/api/play_video/{job_id}")
async def video_stream(job_id: str):
    """
    Stream video file with proper range request support for HTML5 video players.
    Uses FileResponse which automatically handles HTTP range requests correctly.
    Serves the processed video from the outputs directory.
    Automatically adds faststart atom if missing for streaming compatibility.
    """
    video_path = OUTPUTS_DIR / job_id / "processed.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")

    # Check if faststart marker file exists (indicates faststart was already added)
    faststart_marker = video_path.with_suffix(".faststart")

    # If faststart hasn't been added yet, add it now
    if not faststart_marker.exists():
        try:
            await add_faststart_to_video(video_path)
            # Create marker file to indicate faststart was added
            faststart_marker.touch()
        except Exception as e:
            # If faststart addition fails, still try to serve the video
            print(f"Warning: Could not add faststart to {video_path}: {e}")

    # FileResponse automatically handles range requests, Content-Range headers, etc.
    # This is the recommended FastAPI way for serving files with range support
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files.
    """
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_manager.delete_job(job_id)

    return JSONResponse(
        status_code=204, content={"message": "Job deleted successfully"}
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health and system status.
    """
    gpu_available = torch.cuda.is_available()
    model_loaded = gaze_service.gaze_detector is not None

    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        model_loaded=model_loaded,
        storage_path=str(UPLOADS_DIR.parent),
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Gaze Estimation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }

@app.get("/api/progress")
async def progress(job_id: str = Query(...)) -> StreamingResponse:
    """
    Stream progress of a job.
    
    - **job_id**: ID of the job to stream progress for
    
    Raises:
        HTTPException: If job manager is not initialized
    
    Returns:
        StreamingResponse: Streaming response with progress data
    """
    try:
        if not job_manager:
            raise HTTPException(status_code=500, detail="Job manager not initialized")
            
        return StreamingResponse(
            job_manager.stream_progress(job_id), media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))