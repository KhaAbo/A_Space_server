import sys
import cv2
import logging
import numpy as np
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

from cjm_byte_track.core import BYTETracker
from cjm_byte_track.matching import match_detections_with_tracks

# Add gaze-estimation to path
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent
        / "gaze-estimation-testing-main"
        / "gaze-estimation"
    ),
)

from utils.helpers import get_model, draw_bbox_gaze
import uniface

logging.basicConfig(level=logging.INFO, format="%(message)s")


class GazeEstimationService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = None
        self.gaze_detector = None
        self.current_model = None
        self.idx_tensor = None

    def load_model(self, model_name: str, bins: int, weight_path: str):
        """Load the gaze estimation model (singleton pattern)."""
        if self.gaze_detector is None or self.current_model != model_name:
            try:
                logging.info(f"Loading model: {model_name}")
                self.gaze_detector = get_model(model_name, bins, inference_mode=True)
                state_dict = torch.load(weight_path, map_location=self.device)
                self.gaze_detector.load_state_dict(state_dict)
                self.gaze_detector.to(self.device)
                self.gaze_detector.eval()
                self.current_model = model_name
                self.idx_tensor = torch.arange(
                    bins, device=self.device, dtype=torch.float32
                )
                logging.info("Model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                raise

    def load_face_detector(self):
        """Load face detector (singleton pattern)."""
        if self.face_detector is None:
            logging.info("Loading face detector")
            self.face_detector = uniface.RetinaFace()

    def pre_process(self, image):
        """Preprocess face image for gaze estimation."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(448),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = transform(image)
        image_batch = image.unsqueeze(0)
        return image_batch

    def _compute_default_roi(self, width: int, height: int):
        """Approximate the monitor area located just below a webcam."""
        horizontal_margin = int(width * 0.1)  # leave edges of frame
        top = int(height * 0.35)  # assume webcam sits above the screen area
        bottom = min(height - int(height * 0.05), height)  # keep small footer margin
        left = horizontal_margin
        right = width - horizontal_margin
        return left, top, right, bottom

    @staticmethod
    def _is_point_in_roi(point, roi_bounds):
        """Return True if (x, y) lies within roi_bounds."""
        x, y = point
        left, top, right, bottom = roi_bounds
        return left <= x <= right and top <= y <= bottom

    @staticmethod
    def _draw_roi(frame, roi_bounds, is_active: bool):
        """Visualize ROI and status on the frame."""
        left, top, right, bottom = roi_bounds
        color = (0, 200, 0) if is_active else (0, 0, 255)
        label = "Screen ROI"
        status = "LOOKING" if is_active else "NOT LOOKING"

        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        text_y = max(top - 15, 30)
        cv2.putText(
            frame,
            f"{label}: {status}",
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    def process_video(
        self,
        input_path: str,
        output_path: str,
        model_name: str,
        bins: int,
        binwidth: int,
        angle: int,
        weight_path: str,
        progress_callback=None,
    ):
        """Process video file for gaze estimation.

        Args:
            input_path: Path to input video
            output_path: Path to save processed video
            model_name: Name of the model to use
            bins: Number of bins for gaze estimation
            binwidth: Width of each bin
            angle: Angle offset
            weight_path: Path to model weights
            progress_callback: Optional callback function(total_frames, processed_frames)
        """
        # Ensure models are loaded
        self.load_face_detector()
        self.load_model(model_name, bins, weight_path)

        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        # Setup video writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        roi_bounds = self._compute_default_roi(width, height)

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_fps = fps if fps and fps > 0 else 30.0
        tracker = BYTETracker(
            track_thresh=0.3,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=int(round(effective_fps)),
        )
        track_registry: dict[int, dict] = {}
        frame_stats = []
        total_faces_detected = 0
        total_roi_hits = 0
        peak_roi_hits = 0

        # Call progress callback with total frames
        if progress_callback:
            progress_callback(total_frames, 0)

        gaze_data = []

        with torch.no_grad():
            while True:
                success, frame = cap.read()

                if not success:
                    logging.info("Video processing complete")
                    break

                frame_count += 1
                if frame_count % 30 == 0:
                    logging.info(f"Processing frame {frame_count}/{total_frames}")

                # Report progress callback every frame (can be throttled if needed)
                if (
                    progress_callback and frame_count % 10 == 0
                ):  # Update every 10 frames
                    progress_callback(total_frames, frame_count)

                # Detect faces - new uniface API returns list of dicts
                faces = self.face_detector.detect(frame) or []
                frame_roi_hits = []
                valid_detections = []
                tlbr_boxes = []
                detection_output = []

                for face in faces:
                    bbox = np.array(face["bbox"])  # [x1, y1, x2, y2]
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2
                    face_image = frame[y_min:y_max, x_min:x_max]

                    if face_image is None or face_image.size == 0:
                        continue

                    score = float(
                        face.get("score")
                        or face.get("confidence")
                        or face.get("probability")
                        or 1.0
                    )
                    score = float(np.clip(score, 1e-3, 1.0))

                    valid_detections.append(
                        {
                            "bbox": bbox,
                            "x_center": x_center,
                            "y_center": y_center,
                            "face_image": face_image,
                        }
                    )
                    tlbr_boxes.append([x_min, y_min, x_max, y_max])
                    detection_output.append([x_min, y_min, x_max, y_max, score])

                if detection_output:
                    detection_array = np.asarray(detection_output, dtype=float)
                else:
                    detection_array = np.empty((0, 5), dtype=float)

                tracks = tracker.update(
                    detection_array, (height, width), (height, width)
                )
                track_ids = np.full(len(valid_detections), -1, dtype=int)
                if valid_detections and tracks:
                    track_ids = match_detections_with_tracks(
                        np.asarray(tlbr_boxes, dtype=float),
                        track_ids,
                        tracks,
                    )

                frame_timestamp = frame_count / effective_fps if effective_fps else 0.0

                for det_idx, det in enumerate(valid_detections):
                    bbox = det["bbox"]
                    x_center = det["x_center"]
                    y_center = det["y_center"]
                    face_image = det["face_image"]

                    face_tensor = self.pre_process(face_image)
                    face_tensor = face_tensor.to(self.device)

                    gaze_output = self.gaze_detector(face_tensor)

                    if isinstance(gaze_output, tuple):
                        if len(gaze_output) == 2:
                            pitch, yaw = gaze_output
                        elif len(gaze_output) == 3:
                            pitch, yaw = gaze_output[0], gaze_output[1]
                        else:
                            raise ValueError(
                                f"Unexpected model output: {len(gaze_output)} values"
                            )
                    else:
                        raise ValueError(
                            f"Expected tuple output, got {type(gaze_output)}"
                        )

                    pitch_predicted = F.softmax(pitch, dim=1)
                    yaw_predicted = F.softmax(yaw, dim=1)

                    pitch_predicted = (
                        torch.sum(pitch_predicted * self.idx_tensor, dim=1) * binwidth
                        - angle
                    )
                    yaw_predicted = (
                        torch.sum(yaw_predicted * self.idx_tensor, dim=1) * binwidth
                        - angle
                    )

                    pitch_rad = np.radians(pitch_predicted.cpu())
                    yaw_rad = np.radians(yaw_predicted.cpu())

                    draw_bbox_gaze(frame, bbox, pitch_rad, yaw_rad)

                    focal_length = width
                    dx = float(-focal_length * np.tan(pitch_rad))
                    dy = float(-focal_length * np.tan(yaw_rad) / np.cos(pitch_rad))

                    gazepoint_x = int(x_center + dx)
                    gazepoint_y = int(y_center + dy)
                    in_roi = self._is_point_in_roi(
                        (gazepoint_x, gazepoint_y), roi_bounds
                    )
                    frame_roi_hits.append(in_roi)

                    track_id_val = (
                        int(track_ids[det_idx]) if det_idx < len(track_ids) else -1
                    )
                    track_id = track_id_val if track_id_val >= 0 else None

                    if track_id is not None:
                        stats = track_registry.setdefault(
                            track_id,
                            {
                                "track_id": track_id,
                                "first_seen_frame": frame_count,
                                "first_seen_ts": frame_timestamp,
                                "last_seen_frame": frame_count,
                                "last_seen_ts": frame_timestamp,
                                "frames_visible": 0,
                                "roi_hits": 0,
                                "total_detections": 0,
                                "ever_in_roi": False,
                            },
                        )
                        stats["last_seen_frame"] = frame_count
                        stats["last_seen_ts"] = frame_timestamp
                        stats["frames_visible"] += 1
                        stats["total_detections"] += 1
                        if in_roi:
                            stats["roi_hits"] += 1
                        stats["ever_in_roi"] = stats["ever_in_roi"] or in_roi

                    dot_color = (0, 220, 0) if in_roi else (0, 0, 255)
                    cv2.circle(frame, (gazepoint_x, gazepoint_y), 10, dot_color, -1)
                    cv2.circle(frame, (gazepoint_x, gazepoint_y), 12, (0, 0, 0), 2)

                    status_segments = []
                    if track_id is not None:
                        status_segments.append(f"ID {track_id}")
                    status_segments.append(
                        "LOOKING AT ROI" if in_roi else "LOOKING AWAY"
                    )
                    status_text = " - ".join(status_segments)
                    text_origin = (int(bbox[0]), max(int(bbox[1]) - 10, 25))
                    cv2.putText(
                        frame,
                        status_text,
                        text_origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        dot_color,
                        2,
                        cv2.LINE_AA,
                    )

                    gaze_data.append(
                        {
                            "frame": frame_count,
                            "timestamp": frame_timestamp,
                            "face_index": track_id if track_id is not None else det_idx,
                            "track_id": track_id,
                            "bbox": [int(x) for x in bbox],
                            "head_pose": [float(yaw_rad), float(pitch_rad)],
                            "gazepoint": [int(gazepoint_x), int(gazepoint_y)],
                            "in_roi": bool(in_roi),
                        }
                    )

                roi_hits = int(sum(frame_roi_hits))
                peak_roi_hits = max(peak_roi_hits, roi_hits)
                total_faces_detected += len(valid_detections)
                total_roi_hits += roi_hits
                active_track_ids = [int(tid) for tid in track_ids if tid >= 0]
                frame_stats.append(
                    {
                        "frame": frame_count,
                        "timestamp": frame_timestamp,
                        "total_faces": len(valid_detections),
                        "roi_hits": roi_hits,
                        "non_roi_faces": max(len(valid_detections) - roi_hits, 0),
                        "active_track_count": len(set(active_track_ids)),
                        "active_track_ids": sorted(set(active_track_ids)),
                    }
                )

                self._draw_roi(frame, roi_bounds, any(frame_roi_hits))

                # Write frame
                out.write(frame)

        # Final progress update
        if progress_callback:
            progress_callback(total_frames, frame_count)

        # Cleanup
        cap.release()
        out.release()

        # Save gaze data
        json_path = Path(output_path).parent / "gaze_data.json"
        total_non_roi = max(total_faces_detected - total_roi_hits, 0)
        roi_engagement_rate = (
            total_roi_hits / total_faces_detected if total_faces_detected else 0.0
        )
        track_summaries = []
        for stats in sorted(track_registry.values(), key=lambda item: item["track_id"]):
            duration_frames = stats["last_seen_frame"] - stats["first_seen_frame"] + 1
            duration_seconds = (
                duration_frames / effective_fps if effective_fps else duration_frames
            )
            track_summaries.append(
                {
                    "track_id": stats["track_id"],
                    "first_seen_frame": stats["first_seen_frame"],
                    "last_seen_frame": stats["last_seen_frame"],
                    "first_seen_timestamp": stats["first_seen_ts"],
                    "last_seen_timestamp": stats["last_seen_ts"],
                    "frames_visible": stats["frames_visible"],
                    "roi_hits": stats["roi_hits"],
                    "ever_in_roi": stats["ever_in_roi"],
                    "roi_engagement_rate": (
                        stats["roi_hits"] / stats["frames_visible"]
                        if stats["frames_visible"]
                        else 0.0
                    ),
                    "duration_seconds": duration_seconds,
                }
            )

        unique_tracks_total = len(track_summaries)
        unique_tracks_roi = sum(1 for stats in track_summaries if stats["ever_in_roi"])
        unique_tracks_non_roi = unique_tracks_total - unique_tracks_roi
        unique_roi_engagement_rate = (
            unique_tracks_roi / unique_tracks_total if unique_tracks_total else 0.0
        )
        gaze_payload = {
            "detections": gaze_data,
            "frame_stats": frame_stats,
            "tracks": track_summaries,
            "summary": {
                "total_frames": frame_count,
                "total_faces_detected": total_faces_detected,
                "total_roi_hits": total_roi_hits,
                "total_non_roi_faces": total_non_roi,
                "roi_engagement_rate": roi_engagement_rate,
                "peak_concurrent_roi_hits": peak_roi_hits,
                "unique_tracks_total": unique_tracks_total,
                "unique_tracks_roi": unique_tracks_roi,
                "unique_tracks_non_roi": unique_tracks_non_roi,
                "unique_roi_engagement_rate": unique_roi_engagement_rate,
            },
        }
        with open(json_path, "w") as f:
            json.dump(gaze_payload, f, indent=2)

        logging.info(f"Output saved to: {output_path}")
        logging.info(f"Gaze data saved to: {json_path}")


# Global service instance
gaze_service = GazeEstimationService()
