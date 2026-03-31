from __future__ import annotations

import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import PoseLandmarker, RunningMode

from pose_utils import (
    NUM_LANDMARKS,
    build_csv_headers,
    build_pose_options,
    draw_pose_landmarks,
    find_input_file,
    make_landmark_row,
)


def main() -> None:
    input_dir = Path("input_vid")
    output_dir = Path("output_vid")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_video_path = find_input_file(input_dir, stem="input")
    output_video_path = output_dir / "output.mp4"
    output_csv_path = output_dir / "output.csv"

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Failed to read video dimensions.")

    target_fps = 30.0
    target_interval_ms = 1000.0 / target_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, target_fps, (width, height))
    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_video_path}")

    options = build_pose_options(RunningMode.VIDEO)
    input_frame_idx = -1
    output_frame_idx = 0

    with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=build_csv_headers())
        csv_writer.writeheader()

        with PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                input_frame_idx += 1
                source_ts_ms = (input_frame_idx / source_fps) * 1000.0

                while (output_frame_idx * target_interval_ms) <= source_ts_ms + 1e-6:
                    output_ts_ms = int(round(output_frame_idx * target_interval_ms))
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, output_ts_ms)

                    landmarks = None
                    if result.pose_landmarks and result.pose_landmarks[0]:
                        landmarks = result.pose_landmarks[0]

                    annotated = frame.copy()
                    draw_pose_landmarks(annotated, landmarks)
                    video_writer.write(annotated)
                    csv_writer.writerow(
                        make_landmark_row(output_ts_ms, output_frame_idx, landmarks, NUM_LANDMARKS)
                    )

                    output_frame_idx += 1

    cap.release()
    video_writer.release()

    if output_frame_idx == 0:
        raise RuntimeError("No frames were processed from the input video.")

    print(f"Input video: {input_video_path}")
    print(f"Output video: {output_video_path}")
    print(f"Output CSV: {output_csv_path}")
    print(f"Output FPS: {target_fps}")
    print(f"Frames written: {output_frame_idx}")


if __name__ == "__main__":
    main()
