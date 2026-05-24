# Facial Recognition + Emotion Detection

Real-time facial recognition and emotion detection using a webcam feed. The application identifies known faces and displays their dominant emotion on the video stream.

## Features

- **Face Recognition** — Identify known faces from reference photos with configurable tolerance
- **Emotion Detection** — Detect dominant emotions (happy, sad, angry, etc.) using DeepFace
- **Real-time Processing** — Live webcam feed with bounding boxes and labels
- **Performance Optimized** — Frames are downscaled for faster detection; emotions are computed every 5 frames; face crops are resized before analysis

## Requirements

- Python 3.12+
- Webcam
- [uv](https://docs.astral.sh/uv/) (recommended package manager)

## Setup

1. Clone the repository and install dependencies:

   ```bash
   uv sync
   ```

2. Add reference photos of people you want to recognize to the `known_faces/` directory. Each file should be named `<person_name>.jpg` (or `.png`/`.jpeg`) and contain a clear photo of a single face:

   ```
   known_faces/
   ├── alice.jpg
   ├── bob.jpg
   └── charlie.png
   ```

   The filename (without extension) is used as the displayed name.

## Usage

Run the application:

```bash
uv run main.py
```

- A webcam window will open showing the live feed with face bounding boxes and labels
- Recognized names appear with their detected emotion in parentheses (e.g., `alice (happy)`)
- Unknown faces are labeled as `Unknown`
- Press **Q** to quit

## Configuration

All settings are centralized in `config.py`.

| Setting | Default | Description |
|---|---|---|
| `SCALE_FACTOR` | `0.25` | Downscale ratio for face detection. Lower = faster but less accurate on small faces |
| `TOLERANCE` | `0.6` | Face matching tolerance. Lower = stricter matching |
| `EMOTION_EVERY_N_FRAMES` | `5` | How often (in frames) to run emotion detection. Higher values improve FPS |
| `EMOTION_INPUT_SIZE` | `224` | Resize face crop to this size before emotion analysis. Smaller = faster |
| `CAMERA_INDEX` | `0` | Webcam device index |
| `WINDOW_NAME` | `"Facial Recognition + Emotion Detection"` | Display window title |
| `KNOWN_FACES_DIR` | `"known_faces"` | Directory containing reference photos |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Project Structure

```
.
├── main.py               # Entry point — webcam loop, setup, logging
├── config.py             # Centralized settings (all tunable parameters)
├── pipeline.py           # ProcessingWorker — threaded frame pipeline + PipelineResult
├── drawing.py            # draw_results — bounding boxes and labels overlay
├── face_recognizer.py    # FaceRecognizer — loads and matches known faces
├── emotion_detector.py   # EmotionDetector — DeepFace emotion analysis with caching
├── known_faces/          # Reference photos for face recognition
└── pyproject.toml        # Project metadata and dependencies
```

## Dependencies

- [face_recognition](https://github.com/ageitgey/face_recognition) — Face detection and encoding
- [opencv-python](https://opencv.org/) — Webcam capture and image processing
- [DeepFace](https://github.com/serengil/deepface) — Emotion detection
- [numpy](https://numpy.org/) — Array operations
