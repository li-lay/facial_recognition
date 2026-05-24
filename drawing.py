import cv2
import numpy as np

from pipeline import PipelineResult


def draw_results(frame: np.ndarray, result: PipelineResult) -> None:
    for (top, right, bottom, left), name, emotion in zip(
        result.face_locations, result.names, result.emotions
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
