# Video Processing Progress Tracking

## Overview

Frame-based progress tracking has been implemented for video processing jobs. This allows real-time monitoring of how many frames have been processed out of the total frames in the uploaded video.

## What's New

### Progress Fields in Job Status

The API now returns three additional fields when querying job status:

- **`total_frames`** (int|null): Total number of frames in the video
  - Populated once video processing starts
  - `null` while job is in `pending` status

- **`processed_frames`** (int): Number of frames processed so far
  - Starts at `0`
  - Updates every 10 frames during processing
  - Final value equals `total_frames` when completed

- **`progress_percentage`** (float|null): Completion percentage (0-100)
  - Calculated as `(processed_frames / total_frames) * 100`
  - Rounded to 2 decimal places
  - `null` until processing begins

## API Response Example

### During Processing

```json
{
  "job_id": "abc123...",
  "status": "processing",
  "filename": "my_video.mp4",
  "model": "resnet50",
  "created_at": "2025-11-18T10:00:00",
  "started_at": "2025-11-18T10:00:05",
  "completed_at": null,
  "error": null,
  "total_frames": 1500,
  "processed_frames": 750,
  "progress_percentage": 50.0
}
```

### Completed

```json
{
  "job_id": "abc123...",
  "status": "completed",
  "filename": "my_video.mp4",
  "model": "resnet50",
  "created_at": "2025-11-18T10:00:00",
  "started_at": "2025-11-18T10:00:05",
  "completed_at": "2025-11-18T10:05:30",
  "error": null,
  "total_frames": 1500,
  "processed_frames": 1500,
  "progress_percentage": 100.0
}
```

## Implementation Details

### Backend Changes

1. **`api/models.py`**: Added progress fields to `JobInfo` model
2. **`api/job_manager.py`**: 
   - Added `update_progress()` method
   - Modified `create_job()` to initialize progress fields
   - Updated `update_job_status()` to set 100% on completion
3. **`api/gaze_service.py`**: 
   - Added `progress_callback` parameter to `process_video()`
   - Progress updates every 10 frames
4. **`api/main.py`**: 
   - Modified `process_video_background()` to pass progress callback
   - Updated job status endpoints to include progress fields

### Update Frequency

Progress is updated every **10 frames** to balance:
- **Real-time feedback**: Users see progress without long delays
- **Performance**: Not saving job state on every single frame
- **Disk I/O**: Reduced writes to `jobs.json`

You can adjust this by changing line 118 in `api/gaze_service.py`:

```python
if progress_callback and frame_count % 10 == 0:  # Change 10 to your preferred value
    progress_callback(total_frames, frame_count)
```

## Usage Examples

### Python Client

```python
import requests
import time

API_URL = "http://localhost:8000"

# Upload video
with open("video.mp4", "rb") as f:
    response = requests.post(f"{API_URL}/api/upload", files={"file": f})
job_id = response.json()["job_id"]

# Monitor progress
while True:
    job = requests.get(f"{API_URL}/api/jobs/{job_id}").json()
    
    if job['status'] == 'processing' and job['progress_percentage']:
        print(f"Progress: {job['progress_percentage']:.1f}% "
              f"({job['processed_frames']}/{job['total_frames']} frames)")
    
    if job['status'] in ['completed', 'failed']:
        break
    
    time.sleep(2)
```

### Bash Script

```bash
JOB_ID="your-job-id"

while true; do
    STATUS=$(curl -s "http://localhost:8000/api/jobs/$JOB_ID")
    
    PROGRESS=$(echo $STATUS | jq -r '.progress_percentage')
    PROCESSED=$(echo $STATUS | jq -r '.processed_frames')
    TOTAL=$(echo $STATUS | jq -r '.total_frames')
    
    if [ "$PROGRESS" != "null" ]; then
        echo "Progress: $PROGRESS% ($PROCESSED/$TOTAL frames)"
    fi
    
    sleep 2
done
```

### JavaScript/Frontend

```javascript
async function monitorProgress(jobId) {
    const interval = setInterval(async () => {
        const response = await fetch(`http://localhost:8000/api/jobs/${jobId}`);
        const job = await response.json();
        
        if (job.status === 'processing' && job.progress_percentage) {
            console.log(`Progress: ${job.progress_percentage.toFixed(1)}% ` +
                       `(${job.processed_frames}/${job.total_frames} frames)`);
            
            // Update UI progress bar
            updateProgressBar(job.progress_percentage);
        }
        
        if (job.status === 'completed' || job.status === 'failed') {
            clearInterval(interval);
            handleCompletion(job);
        }
    }, 2000); // Check every 2 seconds
}
```

## Testing

Run the updated test script to see progress tracking in action:

```bash
cd other_stuff/test_scripts
python test_api.py
```

You'll see output like:

```
3. Checking status with progress tracking...
   Status: processing - 10.5% (150/1428 frames)
   Status: processing - 20.1% (287/1428 frames)
   Status: processing - 30.7% (438/1428 frames)
   ...
   Status: processing - 99.4% (1420/1428 frames)
   Status: completed

   Job Details:
   - Filename: in_train.mp4
   - Model: resnet50
   - Total Frames: 1428
   - Final Progress: 100.0%
```

## Backward Compatibility

✅ **Fully backward compatible**

- Existing API clients will continue to work
- New fields are optional in responses
- Old job records in `jobs.json` will get default values
- No breaking changes to existing endpoints

## Performance Impact

### Minimal Overhead

- Progress updates every 10 frames (not every frame)
- Uses existing `jobs.json` persistence mechanism
- No additional database queries or network calls
- Update frequency is configurable

### Storage

Each job record grows by ~60 bytes:

```json
"total_frames": 1500,
"processed_frames": 1500,
"progress_percentage": 100.0
```

## Benefits

1. **User Experience**: Users can see real-time progress instead of just "processing"
2. **Time Estimation**: Calculate estimated time remaining based on processing speed
3. **Debugging**: Identify if processing is stuck at a specific frame
4. **Transparency**: Users know exactly what's happening
5. **UI/UX**: Easy to build progress bars and visual indicators

## Future Enhancements

Possible improvements:

1. **ETA Calculation**: Add estimated time remaining based on processing speed
2. **Speed Metrics**: Add frames per second (FPS) processing rate
3. **Stage Tracking**: Track initialization, processing, and finalization stages
4. **WebSocket Support**: Push progress updates instead of polling
5. **Batch Progress**: Track progress across multiple videos

## Files Modified

- ✅ `api/models.py` - Added progress fields
- ✅ `api/job_manager.py` - Added progress tracking methods
- ✅ `api/gaze_service.py` - Added progress callback
- ✅ `api/main.py` - Integrated progress updates
- ✅ `API_README.md` - Updated documentation
- ✅ `other_stuff/test_scripts/test_api.py` - Updated test script

## Questions?

For implementation details, see:
- Code: `api/gaze_service.py` lines 64-180
- API Docs: `API_README.md` lines 108-144
- Examples: `other_stuff/test_scripts/test_api.py`

