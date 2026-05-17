import cv2
import face_recognition
import numpy as np

from face_recognizer import FaceRecognizer
from emotion_detector import EmotionDetector

EMOTION_EVERY_N_FRAMES = 5


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

    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        names: list[str] = []
        emotions: list[str | None] = []

        for i, face_encoding in enumerate(face_encodings):
            name, _ = recognizer.recognize(face_encoding)
            names.append(name)

            if frame_count % EMOTION_EVERY_N_FRAMES == 0:
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
                    emotion = emotion_detector.analyze(face_crop)
                    emotions.append(emotion)
                    print(f"  {name}: {emotion}")
                else:
                    emotions.append(None)
            else:
                emotions.append(None)

        scaled_locations = [
            (t * 4, r * 4, b * 4, l * 4) for (t, r, b, l) in face_locations
        ]
        draw_results(frame, scaled_locations, names, emotions)

        cv2.imshow("Facial Recognition + Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
