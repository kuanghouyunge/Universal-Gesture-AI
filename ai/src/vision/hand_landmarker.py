
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


AI_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = AI_DIR / "models" / "hand_landmarker.task"


class HandLandmarker:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"手部检测模型不存在: {MODEL_PATH}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.detector = (
            vision.HandLandmarker.create_from_options(options)
        )

        self.timestamp_ms = 0

    def detect(self, mp_image):
        """检测画面中的手部关键点。"""
        self.timestamp_ms += 33

        return self.detector.detect_for_video(
            mp_image,
            self.timestamp_ms
        )

    def close(self):
        """释放 MediaPipe 检测器资源。"""
        if self.detector is not None:
            self.detector.close()
            self.detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()