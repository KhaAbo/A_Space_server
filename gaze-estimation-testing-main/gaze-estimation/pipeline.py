"""Gaze Estimation Pipeline - Clean pipeline architecture for gaze detection."""

import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from cjm_byte_track.core import BYTETracker
from cjm_byte_track.matching import match_detections_with_tracks

from utils.helpers import get_model, draw_bbox_gaze
from utils.video_utils import (
    compute_roi_bounds,
    is_point_in_roi,
    draw_roi,
    draw_gazepoint,
    draw_face_status,
)
from gaze_config import Config, get_config
import uniface

logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class FrameResult:
    """Result from processing a single frame."""

    frame: np.ndarray  # Annotated frame
    detections: List[Dict[str, Any]] = field(default_factory=list)
    roi_hits: int = 0
    total_faces: int = 0


@dataclass
class TrackStats:
    """Statistics for a tracked face."""

    track_id: int
    first_seen_frame: int
    first_seen_ts: float
    last_seen_frame: int
    last_seen_ts: float
    frames_visible: int = 0
    roi_hits: int = 0
    total_detections: int = 0
    ever_in_roi: bool = False


class GazeEstimationPipeline:
    """Pipeline for gaze estimation with face detection and tracking.

    This class provides a clean interface for:
    - Face detection using RetinaFace
    - Gaze estimation using configurable models
    - Multi-face tracking with BYTETracker
    - ROI-based engagement detection
    """

    def __init__(self, config: Config | None = None):
        """Initialize the pipeline.

        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or get_config()
        self.device = self.config.device()

        # Model state
        self.face_detector = None
        self.gaze_detector = None
        self.current_model: str | None = None
        self.idx_tensor: torch.Tensor | None = None

        # Get config sections
        self._gaze_cfg = self.config.section("gaze_model")
        self._vis_cfg = self.config.section("visualization")
        self._roi_cfg = self.config.section("roi")
        self._tracker_cfg = self.config.section("tracker")

        # Build preprocessing transform
        self._transform = self._build_transform()

    def _build_transform(self) -> transforms.Compose:
        """Build the image preprocessing transform."""
        img_size = self._gaze_cfg.get("image_size", 448)
        img_mean = self._gaze_cfg.get("img_mean", [0.485, 0.456, 0.406])
        img_std = self._gaze_cfg.get("img_std", [0.229, 0.224, 0.225])

        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])

    def load_models(
        self,
        model_name: str | None = None,
        weight_path: str | None = None,
    ) -> None:
        """Load face detector and gaze estimation model.

        Args:
            model_name: Model architecture name. Uses config default if None.
            weight_path: Path to model weights. Uses config default if None.
        """
        self._load_face_detector()
        self._load_gaze_model(model_name, weight_path)

    def _load_face_detector(self) -> None:
        """Load the face detector (singleton pattern)."""
        if self.face_detector is None:
            logging.info("Loading face detector")
            self.face_detector = uniface.RetinaFace()

    def _load_gaze_model(
        self,
        model_name: str | None = None,
        weight_path: str | None = None,
    ) -> None:
        """Load the gaze estimation model.

        Args:
            model_name: Model architecture name.
            weight_path: Path to model weights.
        """
        model_name = model_name or self._gaze_cfg.get("model_name", "resnet50")
        bins = self._gaze_cfg.get("bins", 90)

        if weight_path is None:
            weight_path = str(self.config.path("gaze_weights"))

        if self.gaze_detector is not None and self.current_model == model_name:
            return  # Already loaded

        try:
            logging.info(f"Loading gaze model: {model_name}")
            self.gaze_detector = get_model(model_name, bins, inference_mode=True)
            state_dict = torch.load(weight_path, map_location=self.device)
            self.gaze_detector.load_state_dict(state_dict)
            self.gaze_detector.to(self.device)
            self.gaze_detector.eval()
            self.current_model = model_name
            self.idx_tensor = torch.arange(
                bins, device=self.device, dtype=torch.float32
            )
            logging.info("Gaze model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading gaze model: {e}")
            raise

    def preprocess_face(self, face_image: np.ndarray) -> torch.Tensor:
        """Preprocess a face image for gaze estimation.

        Args:
            face_image: BGR face crop.

        Returns:
            Preprocessed tensor ready for model input.
        """
        image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        tensor = self._transform(image_rgb)
        return tensor.unsqueeze(0)

    def estimate_gaze(
        self,
        face_tensor: torch.Tensor,
    ) -> Tuple[float, float]:
        """Estimate gaze direction for a face.

        Args:
            face_tensor: Preprocessed face tensor.

        Returns:
            Tuple of (pitch_rad, yaw_rad) in radians.
        """
        bins = self._gaze_cfg.get("bins", 90)
        binwidth = self._gaze_cfg.get("binwidth", 4)
        angle = self._gaze_cfg.get("angle", 180)

        face_tensor = face_tensor.to(self.device)
        gaze_output = self.gaze_detector(face_tensor)

        # Handle different model output formats
        if isinstance(gaze_output, tuple):
            if len(gaze_output) == 2:
                pitch, yaw = gaze_output
            elif len(gaze_output) == 3:
                pitch, yaw = gaze_output[0], gaze_output[1]
            else:
                raise ValueError(f"Unexpected model output: {len(gaze_output)} values")
        else:
            raise ValueError(f"Expected tuple output, got {type(gaze_output)}")

        pitch_predicted = F.softmax(pitch, dim=1)
        yaw_predicted = F.softmax(yaw, dim=1)

        pitch_predicted = (
            torch.sum(pitch_predicted * self.idx_tensor, dim=1) * binwidth - angle
        )
        yaw_predicted = (
            torch.sum(yaw_predicted * self.idx_tensor, dim=1) * binwidth - angle
        )

        pitch_rad = float(np.radians(pitch_predicted.cpu()))
        yaw_rad = float(np.radians(yaw_predicted.cpu()))

        return pitch_rad, yaw_rad

    def compute_gazepoint(
        self,
        face_center: Tuple[int, int],
        pitch_rad: float,
        yaw_rad: float,
        focal_length: float,
    ) -> Tuple[int, int]:
        """Compute screen gazepoint from gaze angles.

        Args:
            face_center: (x, y) center of face bounding box.
            pitch_rad: Pitch angle in radians.
            yaw_rad: Yaw angle in radians.
            focal_length: Focal length for projection (typically frame width).

        Returns:
            (x, y) gazepoint coordinates.
        """
        x_center, y_center = face_center
        dx = float(-focal_length * np.tan(pitch_rad))
        dy = float(-focal_length * np.tan(yaw_rad) / np.cos(pitch_rad))

        gazepoint_x = int(x_center + dx)
        gazepoint_y = int(y_center + dy)
        return gazepoint_x, gazepoint_y

    def create_tracker(self, fps: float) -> BYTETracker:
        """Create a new BYTETracker instance.

        Args:
            fps: Frame rate for the tracker.

        Returns:
            Configured BYTETracker instance.
        """
        return BYTETracker(
            track_thresh=self._tracker_cfg.get("track_thresh", 0.3),
            track_buffer=self._tracker_cfg.get("track_buffer", 30),
            match_thresh=self._tracker_cfg.get("match_thresh", 0.8),
            frame_rate=int(round(fps)),
        )

    def get_roi_bounds(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Get ROI bounds for a frame.

        Args:
            width: Frame width.
            height: Frame height.

        Returns:
            (left, top, right, bottom) ROI bounds.
        """
        return compute_roi_bounds(
            width,
            height,
            horizontal_margin=self._roi_cfg.get("horizontal_margin", 0.1),
            top_margin=self._roi_cfg.get("top_margin", 0.35),
            footer_margin=self._roi_cfg.get("footer_margin", 0.05),
        )

    def process_frame(
        self,
        frame: np.ndarray,
        roi_bounds: Tuple[int, int, int, int],
        tracker: BYTETracker | None = None,
        track_registry: Dict[int, TrackStats] | None = None,
        frame_count: int = 0,
        fps: float = 30.0,
    ) -> FrameResult:
        """Process a single frame for gaze estimation.

        Args:
            frame: Input BGR frame.
            roi_bounds: (left, top, right, bottom) ROI boundaries.
            tracker: Optional BYTETracker for multi-face tracking.
            track_registry: Optional dict to accumulate track statistics.
            frame_count: Current frame number.
            fps: Video frame rate.

        Returns:
            FrameResult with annotated frame and detections.
        """
        height, width = frame.shape[:2]
        frame_timestamp = frame_count / fps if fps else 0.0

        # Detect faces
        faces = self.face_detector.detect(frame) or []
        frame_roi_hits = []
        detections = []
        valid_detections = []
        tlbr_boxes = []
        detection_output = []

        # Process each detected face
        for face in faces:
            bbox = np.array(face["bbox"])
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

            valid_detections.append({
                "bbox": bbox,
                "x_center": x_center,
                "y_center": y_center,
                "face_image": face_image,
            })
            tlbr_boxes.append([x_min, y_min, x_max, y_max])
            detection_output.append([x_min, y_min, x_max, y_max, score])

        # Update tracker
        if detection_output:
            detection_array = np.asarray(detection_output, dtype=float)
        else:
            detection_array = np.empty((0, 5), dtype=float)

        track_ids = np.full(len(valid_detections), -1, dtype=int)
        if tracker is not None:
            tracks = tracker.update(detection_array, (height, width), (height, width))
            if valid_detections and tracks:
                track_ids = match_detections_with_tracks(
                    np.asarray(tlbr_boxes, dtype=float),
                    track_ids,
                    tracks,
                )

        # Process each valid detection
        with torch.no_grad():
            for det_idx, det in enumerate(valid_detections):
                bbox = det["bbox"]
                face_center = (det["x_center"], det["y_center"])
                face_image = det["face_image"]

                # Estimate gaze
                face_tensor = self.preprocess_face(face_image)
                pitch_rad, yaw_rad = self.estimate_gaze(face_tensor)

                # Draw gaze vector
                draw_bbox_gaze(frame, bbox, pitch_rad, yaw_rad)

                # Compute gazepoint
                gazepoint = self.compute_gazepoint(
                    face_center, pitch_rad, yaw_rad, float(width)
                )
                in_roi = is_point_in_roi(gazepoint, roi_bounds)
                frame_roi_hits.append(in_roi)

                # Get track ID
                track_id_val = int(track_ids[det_idx]) if det_idx < len(track_ids) else -1
                track_id = track_id_val if track_id_val >= 0 else None

                # Update track registry
                if track_id is not None and track_registry is not None:
                    if track_id not in track_registry:
                        track_registry[track_id] = TrackStats(
                            track_id=track_id,
                            first_seen_frame=frame_count,
                            first_seen_ts=frame_timestamp,
                            last_seen_frame=frame_count,
                            last_seen_ts=frame_timestamp,
                        )
                    stats = track_registry[track_id]
                    stats.last_seen_frame = frame_count
                    stats.last_seen_ts = frame_timestamp
                    stats.frames_visible += 1
                    stats.total_detections += 1
                    if in_roi:
                        stats.roi_hits += 1
                    stats.ever_in_roi = stats.ever_in_roi or in_roi

                # Draw visualizations
                draw_gazepoint(frame, gazepoint, in_roi, self._vis_cfg)
                draw_face_status(frame, bbox, track_id, in_roi, self._vis_cfg)

                # Record detection (only if looking at ROI)
                if in_roi:
                    detections.append({
                        "frame": frame_count,
                        "track_id": track_id,
                        "bbox": [int(x) for x in bbox],
                        "head_pose": [round(yaw_rad, 3), round(pitch_rad, 3)],
                        "gazepoint": list(gazepoint),
                    })

        # Draw ROI
        draw_roi(frame, roi_bounds, any(frame_roi_hits), self._vis_cfg)

        return FrameResult(
            frame=frame,
            detections=detections,
            roi_hits=int(sum(frame_roi_hits)),
            total_faces=len(valid_detections),
        )

