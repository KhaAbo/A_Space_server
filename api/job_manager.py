import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, AsyncGenerator
import asyncio
from api.models import JobStatus, JobInfo
from api.config import JOBS_FILE, UPLOADS_DIR, OUTPUTS_DIR, FILE_RETENTION_HOURS


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.load_jobs()

    def load_jobs(self):
        """Load jobs from JSON file if it exists."""
        if JOBS_FILE.exists():
            try:
                with open(JOBS_FILE, "r") as f:
                    data = json.load(f)
                    self.jobs = data
            except Exception as e:
                print(f"Error loading jobs: {e}")
                self.jobs = {}
        else:
            self.jobs = {}

    def save_jobs(self):
        """Save jobs to JSON file."""
        try:
            with open(JOBS_FILE, "w") as f:
                json.dump(self.jobs, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving jobs: {e}")

    def create_job(self, job_id: str, filename: str, model: str) -> dict:
        """Create a new job."""
        job = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "filename": filename,
            "model": model,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "total_frames": None,
            "processed_frames": 0,
            "progress_percentage": None,
        }
        self.jobs[job_id] = job
        self.save_jobs()
        return job

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get job by ID."""
        return self.jobs.get(job_id)

    def update_job_status(
        self, job_id: str, status: JobStatus, error: Optional[str] = None
    ):
        """Update job status."""
        if job_id not in self.jobs:
            return

        self.jobs[job_id]["status"] = status

        if status == JobStatus.PROCESSING:
            self.jobs[job_id]["started_at"] = datetime.now().isoformat()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
            # Set progress to 100% on completion
            if status == JobStatus.COMPLETED and self.jobs[job_id].get("total_frames"):
                self.jobs[job_id]["progress_percentage"] = 100.0

        if error:
            self.jobs[job_id]["error"] = error

        self.save_jobs()

    def update_progress(
        self,
        job_id: str,
        total_frames: Optional[int] = None,
        processed_frames: Optional[int] = None,
    ):
        """Update job progress information."""
        if job_id not in self.jobs:
            return

        if total_frames is not None:
            self.jobs[job_id]["total_frames"] = total_frames

        if processed_frames is not None:
            self.jobs[job_id]["processed_frames"] = processed_frames

        # Calculate progress percentage
        if (
            self.jobs[job_id].get("total_frames")
            and self.jobs[job_id]["total_frames"] > 0
        ):
            current_frames = self.jobs[job_id]["processed_frames"]
            total = self.jobs[job_id]["total_frames"]
            self.jobs[job_id]["progress_percentage"] = round(
                (current_frames / total) * 100, 2
            )

        self.save_jobs()

    def delete_job(self, job_id: str):
        """Delete job and associated files."""
        if job_id in self.jobs:
            # Delete files
            upload_dir = UPLOADS_DIR / job_id
            output_dir = OUTPUTS_DIR / job_id

            if upload_dir.exists():
                shutil.rmtree(upload_dir, ignore_errors=True)
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

            # Delete job record
            del self.jobs[job_id]
            self.save_jobs()

    def cleanup_old_jobs(self):
        """Remove jobs and files older than FILE_RETENTION_HOURS."""
        cutoff_time = datetime.now() - timedelta(hours=FILE_RETENTION_HOURS)
        jobs_to_delete = []

        for job_id, job in self.jobs.items():
            created_at = datetime.fromisoformat(job["created_at"])
            if created_at < cutoff_time:
                jobs_to_delete.append(job_id)

        for job_id in jobs_to_delete:
            print(f"Cleaning up old job: {job_id}")
            self.delete_job(job_id)

        return len(jobs_to_delete)

    def list_jobs(self) -> list:
        """List all jobs."""
        return list(self.jobs.values())

    async def stream_progress(self, job_id: str) -> AsyncGenerator[str, None]:
        """
        Stream progress of a job in real time for the frontend with JobInfo JSON object every second.
        The stream is closed when the job is completed or failed or if the job is not found.

        Args:
            job_id: The ID of the job to stream progress for

        Returns:
            AsyncGenerator[str, None]: A generator that yields the progress of the job in real time

        Raises:
            HTTPException: If the job is not found
        """
        while True:
            job = self.jobs.get(job_id)
            if job:
                yield f"data: {json.dumps(job, default=str)}\n\n"
                
                if job['status'] in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    break
            
            await asyncio.sleep(1)