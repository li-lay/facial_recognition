import logging
import threading
from dataclasses import dataclass, field

import cv2
import face_recognition
import numpy as np

from config import CONFIG
from emotion_detector import EmotionDetector
from face_recognizer import FaceRecognizer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    face_locations: list = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    emotions: list[str | None] = field(default_factory=list)


class ProcessingWorker:
    def __init__(
        self, recognizer: FaceRecognizer, emotion_detector: EmotionDetector
    ) -> None:
        self.recognizer = recognizer
        self.emotion_detector = emotion_detector
        self._lock = threading.Lock()
        self._frame_available = threading.Event()
        self._running = True

        self._result = PipelineResult()
        self._pending_frame: np.ndarray | None = None
        self._frame_count = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame.copy()
        self._frame_available.set()

    def get_result(self) -> PipelineResult:
        with self._lock:
            return PipelineResult(
                face_locations=list(self._result.face_locations),
                names=list(self._result.names),
                emotions=list(self._result.emotions),
            )

    def stop(self) -> None:
        self._running = False
        self._frame_available.set()
        self._thread.join()

    def _run(self) -> None:
        while self._running:
            self._frame_available.wait()
            self._frame_available.clear()

            if not self._running:
                break

            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                continue

            self._process_frame(frame)
            self._frame_count += 1

    def _process_frame(self, frame: np.ndarray) -> None:
        sf = CONFIG.SCALE_FACTOR
        usf = CONFIG.UPSCALE_FACTOR

        small_frame = cv2.resize(frame, (0, 0), fx=sf, fy=sf)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)

        if not face_locations:
            with self._lock:
                self._result = PipelineResult()
            return

        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        run_emotion = self._frame_count % CONFIG.EMOTION_EVERY_N_FRAMES == 0

        names: list[str] = []
        emotions: list[str | None] = []

        for i, face_encoding in enumerate(face_encodings):
            name, _ = self.recognizer.recognize(face_encoding, CONFIG.TOLERANCE)
            names.append(name)

            if run_emotion:
                emotion = self._extract_emotion(frame, face_locations[i], name)
            else:
                emotion = self.emotion_detector.get_cached_emotion(name)

            emotions.append(emotion)

        scaled_locations = [
            (t * usf, r * usf, b * usf, l * usf) for (t, r, b, l) in face_locations
        ]

        with self._lock:
            self._result = PipelineResult(
                face_locations=scaled_locations,
                names=names,
                emotions=emotions,
            )

    def _extract_emotion(
        self, frame: np.ndarray, face_location: tuple, name: str
    ) -> str | None:
        t, r, b, l = face_location
        usf = CONFIG.UPSCALE_FACTOR
        top, right = t * usf, r * usf
        bottom, left = b * usf, l * usf

        h, w = frame.shape[:2]
        top = max(0, top)
        right = min(w, right)
        bottom = min(h, bottom)
        left = max(0, left)

        face_crop = frame[top:bottom, left:right]
        if face_crop.size == 0:
            return self.emotion_detector.get_cached_emotion(name)

        face_crop = cv2.resize(
            face_crop, (CONFIG.EMOTION_INPUT_SIZE, CONFIG.EMOTION_INPUT_SIZE)
        )

        emotion = self.emotion_detector.analyze(face_crop, name)
        return emotion or self.emotion_detector.get_cached_emotion(name)
