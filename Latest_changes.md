# Gaze Data API Format

## Overview

The `gaze_data.json` response has been optimized for smaller payload size. The new format reduces file size by **~80%** compared to the original.

## JSON Structure

```json
{
  "meta": {
    "fps": 30.013,
    "total_frames": 561
  },
  "detections": [...],   // Only frames where user is looking at ROI
  "tracks": [...],       // Per-person summary
  "summary": {...}       // Overall statistics
}
```

## Key Changes from Previous Format

| Change                  | Before                    | After                                    |
| ----------------------- | ------------------------- | ---------------------------------------- |
| **Detections filtered** | All frames saved          | Only `in_roi=true` frames saved          |
| **`in_roi` field**      | Present in each detection | Removed (all entries are in ROI)         |
| **`timestamp` field**   | Stored per detection      | Removed (derive from `frame / meta.fps`) |
| **`frame_stats` array** | Included                  | Removed (was redundant)                  |
| **Float precision**     | 15+ decimals              | 3 decimals                               |
| **JSON formatting**     | Indented (pretty)         | Minified (compact)                       |

---

## Detection Object

Each detection represents a single face in a single frame **where the person was looking at the ROI**.

```typescript
interface Detection {
  frame: number; // Frame number (1-indexed)
  track_id: number | null; // Tracked person ID (null if untracked)
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] face bounding box
  head_pose: [number, number]; // [yaw, pitch] in radians
  gazepoint: [number, number]; // [x, y] gaze point in pixels
}
```

### Calculating Timestamp

Since `timestamp` is no longer included, derive it on the frontend:

```javascript
const timestamp = detection.frame / data.meta.fps;
// Example: frame 100 at 30fps = 3.333 seconds
```

---

## Track Object

Each track represents a unique person detected across multiple frames.

```typescript
interface Track {
  track_id: number; // Unique person identifier
  first_frame: number; // First frame where person appeared
  last_frame: number; // Last frame where person appeared
  frames_visible: number; // Total frames person was detected
  roi_hits: number; // Frames where person looked at ROI
  ever_in_roi: boolean; // Did person ever look at ROI?
  roi_rate: number; // Engagement rate (0-1)
  duration_sec: number; // Time person was visible (seconds)
}
```

---

## Summary Object

Aggregated statistics for the entire video.

```typescript
interface Summary {
  total_faces: number; // Total face detections across all frames. This logic counts wrong and shouldn't be used for anything
  roi_hits: number; // Total detections looking at ROI
  non_roi: number; // Total detections looking away
  roi_rate: number; // Overall engagement rate (0-1)
  peak_roi: number; // Max concurrent people looking at ROI
  unique_tracks: number; // Total unique people tracked
  unique_roi: number; // People who looked at ROI at least once
  unique_non_roi: number; // People who never looked at ROI
  unique_roi_rate: number; // Rate of people who engaged (0-1)
}
```

---

## Meta Object

Video metadata needed to interpret the data.

```typescript
interface Meta {
  fps: number; // Video frames per second
  total_frames: number; // Total frames in video
}
```

---

## Complete TypeScript Interface

```typescript
interface GazeData {
  meta: {
    fps: number;
    total_frames: number;
  };
  detections: Array<{
    frame: number;
    track_id: number | null;
    bbox: [number, number, number, number];
    head_pose: [number, number];
    gazepoint: [number, number];
  }>;
  tracks: Array<{
    track_id: number;
    first_frame: number;
    last_frame: number;
    frames_visible: number;
    roi_hits: number;
    ever_in_roi: boolean;
    roi_rate: number;
    duration_sec: number;
  }>;
  summary: {
    total_faces: number;
    roi_hits: number;
    non_roi: number;
    roi_rate: number;
    peak_roi: number;
    unique_tracks: number;
    unique_roi: number;
    unique_non_roi: number;
    unique_roi_rate: number;
  };
}
```

---

## Example Response Size

| Video                | Before  | After  | Reduction |
| -------------------- | ------- | ------ | --------- |
| 561 frames, 1 person | ~316 KB | ~50 KB | ~84%      |

---

## Notes

- **Frame numbers are 1-indexed** (first frame is `1`, not `0`)
- **Only ROI-positive detections are included** in the `detections` array
- **Timestamps can be derived** using `frame / meta.fps`
- **Head pose values** are in radians (multiply by `180/π` for degrees)
- **Gaps in frame numbers** indicate frames where the person was not looking at the ROI
