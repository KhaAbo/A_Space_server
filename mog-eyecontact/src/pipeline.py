from pathlib import Path
import numpy as np

from src.models.mogface import MogFaceDetector
from src.models.eye_contact_cnn import EyeContactEstimator
from src.config import Config
from src.utils import project_utils

crop_face = project_utils.crop_face
draw_box = project_utils.draw_box
select_device = project_utils.select_device


class EyeContactPipeline:
    """
    Pipeline for eye contact detection.
    """
    def __init__(self, config: Config, device: str=None):
        self.config = config # configuration object
        device_cfg = config.section("device").get("default") if device is None else device
        self.device = select_device(device_cfg) # device to use for inference
        self.detector = MogFaceDetector(
            model_path=str(config.path("mogface_weights")),
            config=config,
            device=self.device
        )
        self.eye_contact_estimator = EyeContactEstimator(
            model_path=str(config.path("eye_contact_weights")),
            config=config,
            device=self.device
        )
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame: detect faces and estimate eye contact
        This function is used to process a single frame of the video.

        Args:
            frame: numpy array of the frame to process

        Returns:
            numpy array of the processed frame
        """
        detections = self.detector.detect(frame) # detect faces in the frame
        
        for det in detections:
            x1, y1, x2, y2, conf = det # get the bounding box coordinates and confidence score
            bbox = (int(x1), int(y1), int(x2), int(y2)) # convert the bounding box coordinates to integers
            face_crop = crop_face(frame, bbox) # crop the face from the frame
            if face_crop is None or face_crop.size == 0: # check if the face crop is valid
                continue # skip the detection if the face crop is invalid
            
            # Estimate eye contact score for the face
            eye_contact_score = self.eye_contact_estimator.estimate(face_crop)
            
            # Draw bounding box and score on the frame
            draw_box(frame, bbox, eye_contact_score, float(conf))
                
        return frame
