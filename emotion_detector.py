import cv2
import numpy as np
from deepface import DeepFace


class EmotionDetector:
    def __init__(self):
        self._backend = "opencv"

    def analyze(self, face_img: np.ndarray):
        try:
            result = DeepFace.analyze(
                img_path=face_img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=self._backend,
            )
            if isinstance(result, list):
                result = result[0]
            return result.get("dominant_emotion")
        except Exception:
            return None