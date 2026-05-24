import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    SCALE_FACTOR: float = 0.25
    TOLERANCE: float = 0.6
    EMOTION_EVERY_N_FRAMES: int = 5
    EMOTION_INPUT_SIZE: int = 224
    CAMERA_INDEX: int = 0
    WINDOW_NAME: str = "Facial Recognition + Emotion Detection"
    KNOWN_FACES_DIR: str = "known_faces"
    LOG_LEVEL: int = logging.INFO

    @property
    def UPSCALE_FACTOR(self) -> int:
        return int(1 / self.SCALE_FACTOR)


CONFIG = Settings()
