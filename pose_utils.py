from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarkerOptions, pose_landmarker


MODEL_PATH = Path("models/pose_landmarker_full.task")
POSE_CONNECTIONS = pose_landmarker.PoseLandmarksConnections.POSE_LANDMARKS
NUM_LANDMARKS = len(pose_landmarker.PoseLandmark)


def ensure_model_path(model_path: Path = MODEL_PATH) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please download a pose landmarker .task model and place it there."
        )
    return model_path


def build_pose_options(running_mode, model_path: Path = MODEL_PATH, **kwargs) -> PoseLandmarkerOptions:
    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model_path(model_path))),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        **kwargs,
    )


def find_input_file(input_dir: Path, stem: str = "input") -> Path:
    matches = sorted(path for path in input_dir.glob(f"{stem}.*") if path.is_file())
    if not matches:
        raise FileNotFoundError(f"Input file not found: {input_dir / (stem + '.*')}")
    return matches[0]


def build_csv_headers(num_landmarks: int = NUM_LANDMARKS) -> list[str]:
    headers = ["timestamp_ms", "frame"]
    for idx in range(num_landmarks):
        headers.extend([f"{idx}_x", f"{idx}_y", f"{idx}_z"])
    return headers


def make_landmark_row(
    timestamp_ms: int | None,
    frame_idx: int | None,
    landmarks,
    num_landmarks: int = NUM_LANDMARKS,
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    if timestamp_ms is not None:
        row["timestamp_ms"] = timestamp_ms
    if frame_idx is not None:
        row["frame"] = frame_idx

    for idx in range(num_landmarks):
        row[f"{idx}_x"] = ""
        row[f"{idx}_y"] = ""
        row[f"{idx}_z"] = ""

    if landmarks:
        for idx, landmark in enumerate(landmarks):
            row[f"{idx}_x"] = landmark.x
            row[f"{idx}_y"] = landmark.y
            row[f"{idx}_z"] = landmark.z

    return row


def save_landmarks_csv(csv_path: Path, landmarks) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=build_csv_headers())
        writer.writeheader()
        writer.writerow(make_landmark_row(timestamp_ms=0, frame_idx=0, landmarks=landmarks))


def draw_pose_landmarks(image, landmarks) -> None:
    if not landmarks:
        return

    height, width = image.shape[:2]

    for connection in POSE_CONNECTIONS:
        start = landmarks[connection.start]
        end = landmarks[connection.end]
        x1, y1 = int(start.x * width), int(start.y * height)
        x2, y2 = int(end.x * width), int(end.y * height)
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for landmark in landmarks:
        x, y = int(landmark.x * width), int(landmark.y * height)
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
