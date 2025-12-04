import sys
import cv2
import logging
import numpy as np
import json
from pathlib import Path

import torch
from PIL import Image
import uniface

# Add gazelle to path
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent
        / "gaze-estimation-testing-main"
    
    ),
)

from gazelle.models import get_gazelle_model

logging.basicConfig(level=logging.INFO, format="%(message)s")


class GazelleService:
    """Service for Gazelle gaze estimation."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None
        self.current_model = None
        self.face_detector = uniface.RetinaFace()

    def load_model(self, weight_path: str):
        """Load the Gazelle model (singleton pattern)."""
        if self.model is None:
            try:
                logging.info("Loading Gazelle model...")
                
                # Create model
                self.model, self.transform = get_gazelle_model('gazelle_dinov2_vitl14_inout')
                
                # Load weights
                state_dict = torch.load(weight_path, map_location=self.device)
                self.model.load_gazelle_state_dict(state_dict)
                
                self.model.to(self.device)
                self.model.eval()
                self.current_model = "gazelle"
                logging.info("Gazelle model loaded successfully")
            except Exception as e:
                logging.error(f"Error loading Gazelle model: {e}")
                raise

    def _compute_default_roi(self, width: int, height: int):
        """Approximate the monitor area located just below a webcam."""
        horizontal_margin = int(width * 0.1)
        top = int(height * 0.35)
        bottom = min(height - int(height * 0.05), height)
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

    def heatmap_to_gazepoint(self, heatmap, width, height):
        """Convert heatmap to gazepoint coordinates."""
        heatmap_np = heatmap.detach().cpu().numpy()
        max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
        gaze_x = int((max_index[1] / heatmap_np.shape[1]) * width)
        gaze_y = int((max_index[0] / heatmap_np.shape[0]) * height)
        return gaze_x, gaze_y

    def process_video(
        self,
        input_path: str,
        output_path: str,
        weight_path: str,
        progress_callback=None,
    ):
        """Process video file for gaze estimation using Gazelle."""
        # Ensure model is loaded
        self.load_model(weight_path)

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

                # Report progress callback every 10 frames
                if progress_callback and frame_count % 10 == 0:
                    progress_callback(total_frames, frame_count)

                # Convert BGR to RGB for face detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces using UniFace
                faces = self.face_detector.detect(frame)
                frame_roi_hits = []
                if len(faces) > 0:
                    bboxes = [face['bbox'] for face in faces]  # [x1, y1, x2, y2]
                    norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height])
                                   for bbox in bboxes]]

                    # Prepare input for Gazelle
                    pil_image = Image.fromarray(frame_rgb)
                    img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                    
                    input_data = {
                        "images": img_tensor,
                        "bboxes": norm_bboxes
                    }

                    # Get model predictions
                    output = self.model(input_data)
                    
                    heatmaps = output['heatmap'][0]
                    inout_scores = output['inout'][0] if output['inout'] is not None else None

                    # Process each detected face
                    for i, bbox in enumerate(bboxes):
                        x_min, y_min, x_max, y_max = map(int, bbox[:4])
                        x_center = (x_min + x_max) // 2
                        y_center = (y_min + y_max) // 2

                        # Get inout score
                        inout_score = inout_scores[i].item() if inout_scores is not None else 1.0

                        # Draw face bbox
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        
                        # Display inout score
                        cv2.putText(
                            frame,
                            f"in-frame: {inout_score:.2f}",
                            (x_min, y_max + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )

                        # Only process gaze if looking in-frame
                        if inout_score > 0.0:
                            # Convert heatmap to gazepoint
                            heatmap = heatmaps[i]
                            gaze_x, gaze_y = self.heatmap_to_gazepoint(heatmap, width, height)

                            # Check if in ROI
                            in_roi = self._is_point_in_roi((gaze_x, gaze_y), roi_bounds)
                            frame_roi_hits.append(in_roi)

                            # Draw gaze line
                            cv2.line(frame, (x_center, y_center), (gaze_x, gaze_y), (0, 255, 0), 2)

                            # Draw gazepoint
                            dot_color = (0, 220, 0) if in_roi else (0, 0, 255)
                            cv2.circle(frame, (gaze_x, gaze_y), 10, dot_color, -1)
                            cv2.circle(frame, (gaze_x, gaze_y), 12, (0, 0, 0), 2)

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
                                    "face_index": i,
                                    "bbox": [int(x) for x in bbox],
                                    "gazepoint": [int(gaze_x), int(gaze_y)],
                                    "in_roi": bool(in_roi),
                                    "inout_score": float(inout_score),
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
gazelle_service = GazelleService()