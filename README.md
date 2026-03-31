# thesis-code-mediapipe
Code for deploying MediaPipe using Python

- This repository contains the Python code used to analyze videos with MediaPipe. 
- The code was used in Master’s Thesis: A Systematic Comparison of Microsoft Azure Kinect and Open-Source Motion Capture Tools.

## Files

1. 'video.py': main script for the videos analysis using MediaPipe. 
2. 'pose_utils.py': utility functions for model loading, landmark drawing, input handling, and CSV export.

## Requirements

1. Python 3.10+
2. mediapipe
3. opencv-python

## Model File

- pose_landmarker_full.task was used in the the experiment of the study, it can be downloaded in https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- The model is located in:
  ```text
  models/pose_landmarker_full.task

## video.py

- 'video.py' uses OpenCV to read the input video frame by frame, and uses MediaPipe PoseLandmarker to detect human pose landmarks in each frame. 
It then draws the detected landmarks and skeleton connections on the video frames, saves the video with landmarks and skeleton connections as mp4 video, and exports the landmark coordinates of each frame in csv format.

- The input video should named in 'input.*', it supports common formats that can be opened by OpenCV, such as .mp4, .avi, and .mov.
- The input video is located in：
  ```text
  input_vid/input.mp4
  
- The output video and csv files are in:
  ```text
  output_vid/output.mp4
  output_vid/output.csv
  
- Configuration options of MediaPipe PoseLandmarker used in the experiment of the study:
  ```python
  running_mode=VIDEO,
  num_poses=1,
  min_pose_detection_confidence=0.5,
  min_pose_presence_confidence=0.5,
  min_tracking_confidence=0.5,

## pose_utils.py 

- 'pose_utils.py' provides helper functions used by 'video.py'. 
- It uses MediaPipe Tasks to load the pose landmarker full model, defines the pose landmark connections, prepares CSV headers and rows, searches for the input file, and uses OpenCV drawing functions to render landmarks and skeleton lines on the output video.




