import threading

import cv2
import face_recognition
import numpy as np

from emotion_detector import EmotionDetector
from face_recognizer import FaceRecognizer


def draw_results(
    frame: np.ndarray,
    face_locations: list,
    names: list[str],
    emotions: list[str | None],
) -> None:
    for (top, right, bottom, left), name, emotion in zip(
        face_locations, names, emotions
    ):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        label = name
        if emotion:
            label += f" ({emotion})"

        cv2.rectangle(
            frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED
        )
        cv2.putText(
            frame,
            label,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (0, 0, 0),
            1,
        )


class ProcessingWorker:
    def __init__(
        self, recognizer: FaceRecognizer, emotion_detector: EmotionDetector
    ) -> None:
        self.recognizer = recognizer
        self.emotion_detector = emotion_detector
        self._lock = threading.Lock()
        self._frame_available = threading.Event()
        self._running = True

        self.face_locations: list = []
        self.names: list[str] = []
        self.emotions: list[str | None] = []

        self._pending_frame: np.ndarray | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self._pending_frame = frame.copy()
        self._frame_available.set()

    def get_results(self) -> tuple[list, list[str], list[str | None]]:
        with self._lock:
            return (
                list(self.face_locations),
                list(self.names),
                list(self.emotions),
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

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations
            )

            names: list[str] = []
            emotions: list[str | None] = []

            for i, face_encoding in enumerate(face_encodings):
                name, _ = self.recognizer.recognize(face_encoding)
                names.append(name)

                t, r, b, l = face_locations[i]
                top, right = t * 4, r * 4
                bottom, left = b * 4, l * 4

                h, w = frame.shape[:2]
                top = max(0, top)
                right = min(w, right)
                bottom = min(h, bottom)
                left = max(0, left)

                face_crop = frame[top:bottom, left:right]
                if face_crop.size > 0:
                    emotion = self.emotion_detector.analyze(face_crop, name)
                    emotions.append(
                        emotion or self.emotion_detector.get_cached_emotion(name)
                    )
                else:
                    emotions.append(self.emotion_detector.get_cached_emotion(name))

            scaled_locations = [
                (t * 4, r * 4, b * 4, l * 4) for (t, r, b, l) in face_locations
            ]

            with self._lock:
                self.face_locations = scaled_locations
                self.names = names
                self.emotions = emotions


def main() -> None:
    print("Loading known faces...")
    recognizer = FaceRecognizer("known_faces")

    print("\nInitializing emotion detector...")
    emotion_detector = EmotionDetector()

    print("Starting webcam...")
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam")
        return

    worker = ProcessingWorker(recognizer, emotion_detector)

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Could not read frame")
                break

            worker.submit_frame(frame)

            face_locations, names, emotions = worker.get_results()
            draw_results(frame, face_locations, names, emotions)

            cv2.imshow("Facial Recognition + Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        worker.stop()
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()