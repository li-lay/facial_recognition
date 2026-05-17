import numpy as np
from deepface import DeepFace


class EmotionDetector:
    def __init__(self):
        self._backend = "skip"
        self.last_emotions: dict[str, str] = {}

    def analyze(self, face_img: np.ndarray, name: str = "Unknown") -> str | None:
        try:
            result = DeepFace.analyze(
                img_path=face_img,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=self._backend,
            )
            if isinstance(result, list):
                result = result[0]
            emotion = result.get("dominant_emotion")
            if emotion and name != "Unknown":
                self.last_emotions[name] = emotion
            return emotion
        except Exception:
            return None

    def get_cached_emotion(self, name: str) -> str | None:
        return self.last_emotions.get(name)