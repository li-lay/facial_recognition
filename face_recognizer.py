import logging
import os

import face_recognition

from config import CONFIG

logger = logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(self, known_faces_dir: str = CONFIG.KNOWN_FACES_DIR):
        self.known_encodings: list = []
        self.known_names: list = []
        self.load_known_faces(known_faces_dir)

    def load_known_faces(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info("Created directory: %s", directory)
            logger.info("Place reference photos named as <person_name>.jpg")
            return

        for filename in os.listdir(directory):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            path = os.path.join(directory, filename)
            name = os.path.splitext(filename)[0]

            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                self.known_encodings.append(encodings[0])
                self.known_names.append(name)
                logger.info("  Loaded: %s", name)
            else:
                logger.warning("  No face found in %s", filename)

        logger.info("  Total known faces: %d", len(self.known_names))

    def recognize(self, face_encoding, tolerance: float = CONFIG.TOLERANCE):
        if not self.known_encodings or not self.known_names:
            return "Unknown", None

        distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        best_match_index = int(distances.argmin())

        if distances[best_match_index] < tolerance:
            return self.known_names[best_match_index], distances[best_match_index]

        return "Unknown", None
