import logging

import cv2

from config import CONFIG
from drawing import draw_results
from emotion_detector import EmotionDetector
from face_recognizer import FaceRecognizer
from pipeline import ProcessingWorker

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=CONFIG.LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logging()

    logger.info("Loading known faces from %s ...", CONFIG.KNOWN_FACES_DIR)
    recognizer = FaceRecognizer(CONFIG.KNOWN_FACES_DIR)

    logger.info("Initializing emotion detector...")
    emotion_detector = EmotionDetector()

    logger.info("Opening webcam (index %d)...", CONFIG.CAMERA_INDEX)
    video_capture = cv2.VideoCapture(CONFIG.CAMERA_INDEX)

    if not video_capture.isOpened():
        logger.error("Could not open webcam at index %d", CONFIG.CAMERA_INDEX)
        return

    worker = ProcessingWorker(recognizer, emotion_detector)

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                logger.error("Failed to read frame from webcam")
                break

            worker.submit_frame(frame)
            result = worker.get_result()
            draw_results(frame, result)

            cv2.imshow(CONFIG.WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key pressed")
                break
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        worker.stop()
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
