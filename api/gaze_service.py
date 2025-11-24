import sys
import cv2
import logging
import numpy as np
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

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
                faces = self.face_detector.detect(frame)
                frame_roi_hits = []

                # Process each detected face
                for face in faces:
                    bbox = np.array(face["bbox"])  # [x1, y1, x2, y2]
                    keypoint = face["landmarks"]  # [[x1, y1], [x2, y2], ...]
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2

                    # Extract face region
                    face_image = frame[y_min:y_max, x_min:x_max]

                    # Skip if cropped image is empty or invalid
                    if face_image is None or face_image.size == 0:
                        continue

                    # Preprocess
                    face_tensor = self.pre_process(face_image)
                    face_tensor = face_tensor.to(self.device)

                    # Predict gaze
                    gaze_output = self.gaze_detector(face_tensor)

                    # Handle different return formats
                    if isinstance(gaze_output, tuple):
                        if len(gaze_output) == 2:
                            pitch, yaw = gaze_output
                        elif len(gaze_output) == 3:
                            # Some models might return 3 values, take first 2
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

                    # Map from bins to angles
                    pitch_predicted = (
                        torch.sum(pitch_predicted * self.idx_tensor, dim=1) * binwidth
                        - angle
                    )
                    yaw_predicted = (
                        torch.sum(yaw_predicted * self.idx_tensor, dim=1) * binwidth
                        - angle
                    )

                    # Convert to radians
                    pitch_rad = np.radians(pitch_predicted.cpu())
                    yaw_rad = np.radians(yaw_predicted.cpu())

                    # Draw bbox and gaze arrow
                    draw_bbox_gaze(frame, bbox, pitch_rad, yaw_rad)

                    # Calculate gazepoint
                    # NOTE: Analysis of draw_gaze and gaze_to_3d reveals that:
                    # pitch_rad variable actually contains Yaw (horizontal angle)
                    # yaw_rad variable actually contains Pitch (vertical angle)

                    # Assume focal length is roughly equal to image width
                    focal_length = width

                    # Project gaze to 2D plane using pinhole camera model
                    # x = -f * tan(Yaw)
                    # y = -f * tan(Pitch) / cos(Yaw)

                    dx = -focal_length * np.tan(pitch_rad)
                    dy = -focal_length * np.tan(yaw_rad) / np.cos(pitch_rad)

                    gazepoint_x = int(x_center + dx)
                    gazepoint_y = int(y_center + dy)
                    in_roi = self._is_point_in_roi(
                        (gazepoint_x, gazepoint_y), roi_bounds
                    )
                    frame_roi_hits.append(in_roi)

                    # Draw gazepoint
                    dot_color = (0, 220, 0) if in_roi else (0, 0, 255)
                    cv2.circle(frame, (gazepoint_x, gazepoint_y), 10, dot_color, -1)
                    cv2.circle(frame, (gazepoint_x, gazepoint_y), 12, (0, 0, 0), 2)

                    status_text = "LOOKING AT ROI" if in_roi else "LOOKING AWAY"
                    text_origin = (x_min, max(y_min - 10, 25))
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

                    # Store data
                    gaze_data.append(
                        {
                            "frame": frame_count,
                            "timestamp": frame_count / fps,
                            "face_index": 0,
                            "bbox": [int(x) for x in bbox],
                            # Store as [Pitch, Yaw] for standard convention
                            # yaw_rad is Pitch, pitch_rad is Yaw
                            "head_pose": [float(yaw_rad), float(pitch_rad)],
                            "gazepoint": [int(gazepoint_x), int(gazepoint_y)],
                            "in_roi": bool(in_roi),
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
        with open(json_path, "w") as f:
            json.dump(gaze_data, f, indent=2)

        logging.info(f"Output saved to: {output_path}")
        logging.info(f"Gaze data saved to: {json_path}")


# Global service instance
gaze_service = GazeEstimationService()
