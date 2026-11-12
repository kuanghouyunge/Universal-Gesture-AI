
from pathlib import Path

import joblib
import numpy as np

from src.vision.hand_features import (
    normalize_landmarks,
    extract_joint_angles,
)

AI_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = AI_DIR / "models" / "gesture_classifier.pkl"

GESTURE_LABELS = {
    0: "OPEN_PALM",
    1: "FIST",
    2: "POINT",
    3: "THUMBS_UP",
}


class GestureRecognizer:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {MODEL_PATH}"
            )

        self.model = joblib.load(MODEL_PATH)

        if self.model.n_features_in_ != 68:
            raise ValueError(
                f"模型需要 68 维特征，"
                f"当前为 {self.model.n_features_in_} 维"
            )

    def predict(self, landmarks):
        """识别一只手的手势。"""
        
        coordinates = normalize_landmarks(landmarks)
        angles = extract_joint_angles(landmarks)

        features = np.concatenate(
            [coordinates, angles]
        ).reshape(1, -1)

        prediction = int(self.model.predict(features)[0])

        probabilities = self.model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

        probability_text = {
            GESTURE_LABELS[int(class_id)]: round(float(prob), 3)
            for class_id, prob in zip(
                self.model.classes_,
                probabilities
            )
        }

        return {
            "gesture_id": prediction,
            "gesture": GESTURE_LABELS[prediction],
            "confidence": confidence,
            "probabilities": probability_text,
        }
