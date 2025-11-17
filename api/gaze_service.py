import sys
import cv2
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

# Add gaze-estimation to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gaze-estimation-testing-main" / "gaze-estimation"))

from utils.helpers import get_model, draw_bbox_gaze
import uniface

logging.basicConfig(level=logging.INFO, format='%(message)s')


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
                self.idx_tensor = torch.arange(bins, device=self.device, dtype=torch.float32)
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
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        image_batch = image.unsqueeze(0)
        return image_batch
    
    def process_video(self, input_path: str, output_path: str, model_name: str, 
                      bins: int, binwidth: int, angle: int, weight_path: str):
        """Process video file for gaze estimation."""
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
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with torch.no_grad():
            while True:
                success, frame = cap.read()
                
                if not success:
                    logging.info("Video processing complete")
                    break
                
                frame_count += 1
                if frame_count % 30 == 0:
                    logging.info(f"Processing frame {frame_count}/{total_frames}")
                
                # Detect faces - new uniface API returns list of dicts
                faces = self.face_detector.detect(frame)
                
                # Process each detected face
                for face in faces:
                    bbox = np.array(face['bbox'])  # [x1, y1, x2, y2]
                    keypoint = face['landmarks']  # [[x1, y1], [x2, y2], ...]
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    
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
                            raise ValueError(f"Unexpected model output: {len(gaze_output)} values")
                    else:
                        raise ValueError(f"Expected tuple output, got {type(gaze_output)}")
                    
                    pitch_predicted = F.softmax(pitch, dim=1)
                    yaw_predicted = F.softmax(yaw, dim=1)
                    
                    # Map from bins to angles
                    pitch_predicted = torch.sum(pitch_predicted * self.idx_tensor, dim=1) * binwidth - angle
                    yaw_predicted = torch.sum(yaw_predicted * self.idx_tensor, dim=1) * binwidth - angle
                    
                    # Convert to radians
                    pitch_rad = np.radians(pitch_predicted.cpu())
                    yaw_rad = np.radians(yaw_predicted.cpu())
                    
                    # Draw bbox and gaze arrow
                    draw_bbox_gaze(frame, bbox, pitch_rad, yaw_rad)
                
                # Write frame
                out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        logging.info(f"Output saved to: {output_path}")


# Global service instance
gaze_service = GazeEstimationService()