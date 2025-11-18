import httpx
from datetime import datetime
from typing import Optional
from api.config import DISCORD_WEBHOOK_URL, API_BASE_URL


async def send_job_notification(job_data: dict) -> None:
    """
    Send a Discord webhook notification when a job completes or fails.
    
    Args:
        job_data: Dictionary containing job information with keys:
            - job_id: str
            - status: str (completed or failed)
            - filename: str
            - model: str
            - created_at: str (ISO format)
            - started_at: Optional[str] (ISO format)
            - completed_at: Optional[str] (ISO format)
            - error: Optional[str]
            - total_frames: Optional[int]
            - processed_frames: Optional[int]
            - progress_percentage: Optional[float]
    """
    # Skip if webhook URL is not configured
    if not DISCORD_WEBHOOK_URL:
        return
    
    try:
        status = job_data.get("status", "").lower()
        
        # Determine embed color and title
        if status == "completed":
            color = 0x00ff00  # Green
            title = "✅ Job Completed"
        elif status == "failed":
            color = 0xff0000  # Red
            title = "❌ Job Failed"
        else:
            # Don't send notification for other statuses
            return
        
        # Parse timestamps
        created_at = datetime.fromisoformat(job_data["created_at"]) if job_data.get("created_at") else None
        started_at = datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None
        completed_at = datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None
        
        # Calculate processing duration
        duration_str = "N/A"
        if started_at and completed_at:
            duration = completed_at - started_at
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            if hours > 0:
                duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                duration_str = f"{minutes}m {seconds}s"
            else:
                duration_str = f"{seconds}s"
        
        # Build embed fields
        fields = [
            {
                "name": "Job ID",
                "value": job_data.get("job_id", "N/A"),
                "inline": True
            },
            {
                "name": "Filename",
                "value": job_data.get("filename", "N/A"),
                "inline": True
            },
            {
                "name": "Model",
                "value": job_data.get("model", "N/A"),
                "inline": True
            }
        ]
        
        # Add processing duration
        fields.append({
            "name": "Processing Duration",
            "value": duration_str,
            "inline": True
        })
        
        # Add progress information if available
        total_frames = job_data.get("total_frames")
        processed_frames = job_data.get("processed_frames")
        progress_percentage = job_data.get("progress_percentage")
        
        if total_frames is not None and processed_frames is not None:
            progress_info = f"{processed_frames}/{total_frames} frames"
            if progress_percentage is not None:
                progress_info += f" ({progress_percentage:.1f}%)"
            fields.append({
                "name": "Progress",
                "value": progress_info,
                "inline": True
            })
        
        # Add completion time
        if completed_at:
            fields.append({
                "name": "Completed At",
                "value": completed_at.strftime("%Y-%m-%d %H:%M:%S"),
                "inline": True
            })
        
        # Add error message if failed
        if status == "failed" and job_data.get("error"):
            fields.append({
                "name": "Error",
                "value": job_data["error"][:1024],  # Discord embed field value limit
                "inline": False
            })
        
        # Add download link for completed jobs
        if status == "completed":
            job_id = job_data.get("job_id", "")
            download_url = f"{API_BASE_URL}/api/download/{job_id}"
            fields.append({
                "name": "Download",
                "value": f"[Download processed video]({download_url})",
                "inline": False
            })
        
        # Create embed
        embed = {
            "title": title,
            "color": color,
            "fields": fields,
            "timestamp": completed_at.isoformat() if completed_at else datetime.now().isoformat()
        }
        
        # Create webhook payload
        payload = {
            "embeds": [embed]
        }
        
        # Send webhook request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                DISCORD_WEBHOOK_URL,
                json=payload,
                timeout=10.0
            )
            response.raise_for_status()
            
    except Exception as e:
        # Log error but don't raise - webhook failures shouldn't affect job processing
        print(f"Error sending Discord webhook: {e}")

