import sys
import json
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from typing import Callable
import cv2

# Add external modules to path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_MODULES = [
    "mobilegaze",
    "mog-eyecontact",
]

for module_path in EXTERNAL_MODULES:
    full_path = str(BACKEND_ROOT / module_path)
    if full_path not in sys.path:
        sys.path.insert(0, full_path)

from utils.helpers import get_model, draw_bbox_gaze
from src.models.mogface import MogFaceDetector
from src.config import Config as MogConfig, load_config
from src.models.eye_contact_cnn import EyeContactEstimator
from gaze_config import Config, get_config
from pipeline import GazeEstimationPipeline, TrackStats
import config as api_config
from .adapter import MogFaceAdapter
import uniface
from enum import Enum
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(message)s")

class FaceDetectorModel(str, Enum):
    RetinaFace = "RetinaFace"
    MogFace = "MogFace"

class GazeDetectorModel(str, Enum):
    MobileGaze = "MobileGaze"
    EyeContactCNN = "EyeContactCNN"

@dataclass
class FaceDetector:
    model_name: FaceDetectorModel
    config_path: str
    device: torch.device

@dataclass
class GazeDetector:
    model_name: GazeDetectorModel
    config_path: str
    device: torch.device

@dataclass
class Pipeline:
    face_detector: FaceDetector
    gaze_detector: GazeDetector

class GazeEstimationService:
    """
    API-compatible service wrapper for gaze estimation.

    This class maintains the original interface while delegating to
    GazeEstimationPipeline for the actual processing.
    """

    def __init__(self):
        self.face_models = [FaceDetectorModel.RetinaFace, FaceDetectorModel.MogFace]
        self.gaze_models = [GazeDetectorModel.MobileGaze, GazeDetectorModel.EyeContactCNN]
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
        self.config = get_config()
        self.pipeline = GazeEstimationPipeline(self.config) # todo: adjust this to both fit MobileGaze and EyeContactCNN



    @property
    def gaze_detector(self):
        """Expose gaze detector for backward compatibility."""
        return self.pipeline.gaze_detector

    @property
    def face_detector(self):
        """Expose face detector for backward compatibility."""
        return self.pipeline.face_detector

    def load_face_detector(self, model_name: FaceDetectorModel) -> FaceDetector:
        """Load the face detector model.

        Args:
            model_name: Name of the model architecture.
        """
        match model_name:
            case FaceDetectorModel.MogFace:
                logging.info("Loading MogFaceDetector")
                config_path = Path(__file__).parent.parent / "mog-eyecontact" / "config" / "config.yml"
                config = load_config(config_path) # load mogface config
            
                detector = MogFaceDetector(
                    model_path=config.path("mogface_weights"),
                    config=config,
                    device=self.device)

                return MogFaceAdapter(detector)
            
            case FaceDetectorModel.RetinaFace:
                logging.info("Loading RetinaFaceDetector")
                return uniface.RetinaFace()
            
            case _:
                logging.warning(f"Unknown face detector model: {model_name}. Defaulting to RetinaFace.")
                return uniface.RetinaFace()

    def load_gaze_detector(self, model_name: GazeDetectorModel) -> GazeDetector:
        """Load the gaze detector model (backward compatibility).

        Args:
            model_name: Name of the model architecture.
        """
        match model_name:
            case GazeDetectorModel.MobileGaze:
                logging.info("Loading MobileGazeDetector")
                self.pipeline._load_gaze_model("resnet50")
                return self.pipeline.gaze_detector

            case GazeDetectorModel.EyeContactCNN:
                logging.info("Loading EyeContactCNNDetector")
                config_path = Path(__file__).parent.parent / "mog-eyecontact" / "config" / "config.yml"
                mog_config = load_config(config_path)
                
                eyecontact_detector = EyeContactEstimator(
                    model_path=str(mog_config.path("eye_contact_weights")),
                    config=mog_config,
                    device=self.device
                )   
                return eyecontact_detector

            case _:
                logging.warning(f"Unknown gaze detector model: {model_name}. Defaulting to MobileGaze (ResNet50).")
                self.pipeline._load_gaze_model("resnet50")
                return self.pipeline.gaze_detector

    def process_video(
        self,
        face_model_name: FaceDetectorModel,
        gaze_model_name: GazeDetectorModel,
        input_path: str,
        output_path: str,
        model_name: str,
        weight_path: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Process video file for gaze estimation.

        Args:
            input_path: Path to input video.
            output_path: Path to save processed video.
            model_name: Name of the model to use.
            weight_path: Path to model weights.
            progress_callback: Optional callback(total_frames, processed_frames).
            face_model_name: Name of face detector model.
            gaze_model_name: Name of gaze detector model.
        """
        # Determine models to use
        face_model = self.load_face_detector(face_model_name)
        gaze_model = self.load_gaze_detector(gaze_model_name)

        if face_model is None or gaze_model is None:
            raise ValueError("Model not found, please check the error log")

        # Set models in pipeline
        self.pipeline.face_detector = face_model
        self.pipeline.gaze_detector = gaze_model


        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_cfg = self.config.section("video")
        effective_fps = fps if fps and fps > 0 else video_cfg.get("default_fps", 30.0)

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*video_cfg.get("output_codec", "mp4v"))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Setup tracking
        roi_bounds = self.pipeline.get_roi_bounds(width, height)
        tracker = self.pipeline.create_tracker(effective_fps)
        track_registry: dict[int, TrackStats] = {}

        # Processing stats
        frame_count = 0
        total_faces_detected = 0
        total_roi_hits = 0
        peak_roi_hits = 0
        gaze_data = []

        # Progress callback intervals
        progress_interval = video_cfg.get("progress_interval", 10)
        log_interval = video_cfg.get("log_interval", 30)

        # Initial progress callback
        if progress_callback:
            progress_callback(total_frames, 0)

        # Process frames
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("Video processing complete")
                break

            frame_count += 1
            if frame_count % log_interval == 0:
                logging.info(f"Processing frame {frame_count}/{total_frames}")

            # Progress callback
            if progress_callback and frame_count % progress_interval == 0:
                progress_callback(total_frames, frame_count)

            # Process frame through pipeline
            result = self.pipeline.process_frame(
                frame=frame,
                roi_bounds=roi_bounds,
                tracker=tracker,
                track_registry=track_registry,
                frame_count=frame_count,
                fps=effective_fps,
            )

            # Accumulate stats
            total_faces_detected += result.total_faces
            total_roi_hits += result.roi_hits
            peak_roi_hits = max(peak_roi_hits, result.roi_hits)
            gaze_data.extend(result.detections)

            # Write frame
            out.write(result.frame)

        # Final progress update
        if progress_callback:
            progress_callback(total_frames, frame_count)

        # Cleanup video
        cap.release()
        out.release()

        # Build and save gaze data JSON
        self._save_gaze_data(
            output_path=output_path,
            gaze_data=gaze_data,
            track_registry=track_registry,
            total_faces_detected=total_faces_detected,
            total_roi_hits=total_roi_hits,
            peak_roi_hits=peak_roi_hits,
            frame_count=frame_count,
            effective_fps=effective_fps,
        )

        logging.info(f"Output saved to: {output_path}")

    def _save_gaze_data(
        self,
        output_path: str,
        gaze_data: list,
        track_registry: dict[int, TrackStats],
        total_faces_detected: int,
        total_roi_hits: int,
        peak_roi_hits: int,
        frame_count: int,
        effective_fps: float,
    ) -> None:
        """Save gaze detection data to JSON file.

        Args:
            output_path: Path to output video (JSON saved alongside).
            gaze_data: List of detection records.
            track_registry: Track statistics registry.
            total_faces_detected: Total face detections across all frames.
            total_roi_hits: Total ROI hits across all frames.
            peak_roi_hits: Maximum ROI hits in a single frame.
            frame_count: Total frames processed.
            effective_fps: Video frame rate.
        """
        json_path = Path(output_path).parent / "gaze_data.json"

        # Build track summaries
        track_summaries = []
        for stats in sorted(track_registry.values(), key=lambda s: s.track_id):
            duration_frames = stats.last_seen_frame - stats.first_seen_frame + 1
            duration_seconds = (
                duration_frames / effective_fps if effective_fps else duration_frames
            )
            track_summaries.append({
                "track_id": stats.track_id,
                "first_frame": stats.first_seen_frame,
                "last_frame": stats.last_seen_frame,
                "frames_visible": stats.frames_visible,
                "roi_hits": stats.roi_hits,
                "ever_in_roi": stats.ever_in_roi,
                "roi_rate": round(
                    stats.roi_hits / stats.frames_visible
                    if stats.frames_visible else 0.0,
                    3,
                ),
                "duration_sec": round(duration_seconds, 3),
            })

        # Compute summary statistics
        total_non_roi = max(total_faces_detected - total_roi_hits, 0)
        roi_engagement_rate = (
            total_roi_hits / total_faces_detected if total_faces_detected else 0.0
        )
        unique_tracks_total = len(track_summaries)
        unique_tracks_roi = sum(1 for s in track_summaries if s["ever_in_roi"])
        unique_tracks_non_roi = unique_tracks_total - unique_tracks_roi
        unique_roi_engagement_rate = (
            unique_tracks_roi / unique_tracks_total if unique_tracks_total else 0.0
        )

        # Build payload
        gaze_payload = {
            "meta": {
                "fps": round(effective_fps, 3),
                "total_frames": frame_count,
            },
            "detections": gaze_data,
            "tracks": track_summaries,
            "summary": {
                "total_faces": total_faces_detected,
                "roi_hits": total_roi_hits,
                "non_roi": total_non_roi,
                "roi_rate": round(roi_engagement_rate, 3),
                "peak_roi": peak_roi_hits,
                "unique_tracks": unique_tracks_total,
                "unique_roi": unique_tracks_roi,
                "unique_non_roi": unique_tracks_non_roi,
                "unique_roi_rate": round(unique_roi_engagement_rate, 3),
            },
        }

        # Write compact JSON
        with open(json_path, "w") as f:
            json.dump(gaze_payload, f, separators=(",", ":"))

        logging.info(f"Gaze data saved to: {json_path}")


# Global service instance (backward compatibility)
gaze_service = GazeEstimationService()
