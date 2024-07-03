import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import argparse
import logging

# Argument parser for customizable input options
parser = argparse.ArgumentParser(description='Hand Detection')
parser.add_argument('--video_source', type=int, default=0, help='Index of video source')
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Camera setup
cap = cv2.VideoCapture(args.video_source)
if not cap.isOpened():
    logging.error("Error: Could not open video capture.")
    exit()

# Frame capture thread function
def capture_frames():
    global ret, frame
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Error: Could not read frame.")
            break

# Initialize the frame and ret variables
ret = False
frame = None

# Start the frame capture thread
frame_thread = threading.Thread(target=capture_frames)
frame_thread.start()

while True:
    if frame is None:
        continue

    start_time = time.time()  # Start time for FPS calculation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow('Hand Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
