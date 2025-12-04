

class MogFaceAdapter:
    """Adapts MogFaceDetector to return list of dicts expected by GazeEstimationService."""
    def __init__(self, detector):
        self.detector = detector

    def detect(self, image):
        # MogFace returns numpy array: [[x1, y1, x2, y2, score], ...]
        detections = self.detector.detect(image)
        results = []
        if detections is not None and len(detections) > 0:
            for det in detections:
                results.append({
                    "bbox": det[:4].tolist(),
                    "score": float(det[4])
                })
        return results
