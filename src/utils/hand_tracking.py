from mediapipe.tasks.python import vision, BaseOptions
from mediapipe import ImageFormat
import mediapipe as mp
import cv2
import time


class HandTracker:
    def __init__(self):
        self._latest_landmarks = None
        self.frame_counter = 0

        base_options = BaseOptions(model_asset_path="src/assets/hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            result_callback=self._on_result,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _on_result(
        self,
        result: vision.HandLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        if result.hand_landmarks:
            self._latest_landmarks = result.hand_landmarks[0]
        else:
            self._latest_landmarks = None

    def process_frame(self, frame):
        mp_image = mp.Image(image_format=ImageFormat.SRGB, data=frame)
        timestamp_ms = int(time.time() * 1000)
        self.detector.detect_async(mp_image, timestamp_ms)

    def draw_landmarks(self, frame, landmarks):
        if not landmarks:
            return
        h, w, _ = frame.shape
        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            start = landmarks[connection[0]]
            end = landmarks[connection[1]]
            x0, y0 = int(start.x * w), int(start.y * h)
            x1, y1 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    def is_hand_open(self, landmarks):
        return all(
            [
                landmarks[8].y < landmarks[6].y,
                landmarks[12].y < landmarks[10].y,
                landmarks[16].y < landmarks[14].y,
                landmarks[20].y < landmarks[18].y,
            ]
        )

    def get_latest_landmarks(self):
        return self._latest_landmarks
