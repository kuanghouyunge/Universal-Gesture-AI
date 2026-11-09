import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesture_labels import GESTURE_LABELS

# 导入现有特征提取函数
AI_DIR = Path(__file__).resolve().parents[2]
VISION_DIR = AI_DIR / "src" / "vision"
sys.path.insert(0, str(VISION_DIR))

from hand_features import normalize_landmarks, extract_joint_angles


MODEL_PATH = AI_DIR / "models" / "hand_landmarker.task"
DATA_PATH = AI_DIR / "datasets" / "gestures" / "gesture_data.csv"

FEATURE_COLUMNS = (
    [f"feature_{i}" for i in range(63)]
    + [f"angle_{i}" for i in range(5)]
)

HEADER = [
    "label_id",
    "label",
    "handedness",
] + FEATURE_COLUMNS


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    selected_id = None
    timestamp_ms = 0

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )

    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms += 33
                results = detector.detect_for_video(image, timestamp_ms)

                key = cv2.waitKey(1) & 0xFF

                # 选择手势标签
                if key in [ord(str(i)) for i in GESTURE_LABELS]:
                    selected_id = int(chr(key))
                    print("当前标签：", GESTURE_LABELS[selected_id])

                # 显示当前状态
                if selected_id is None:
                    status = "Select gesture: 0-4"
                else:
                    status = f"Gesture: {GESTURE_LABELS[selected_id]}"

                cv2.putText(
                    frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2
                )

                if results.hand_landmarks:
                    landmarks = results.hand_landmarks[0]

                    features = normalize_landmarks(landmarks)
                    angles = extract_joint_angles(landmarks)

                    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        h, w, _ = frame.shape
                        points = [
                            (int(lm.x * w), int(lm.y * h))
                            for lm in landmarks
                        ]
                        cv2.line(
                            frame,
                            points[connection.start],
                            points[connection.end],
                            (255, 0, 0), 2
                        )

                    if key == ord(" ") and selected_id is not None:
                        raw_label = (
                            results.handedness[0][0].category_name
                        )

                        handedness = (
                            "Right" if raw_label == "Left" else "Left"
                        )

                        row = [
                            selected_id,
                            GESTURE_LABELS[selected_id],
                            handedness,
                        ] + features.tolist() + angles.tolist()

                        file_exists = DATA_PATH.exists()

                        with open(
                            DATA_PATH, "a",
                            newline="", encoding="utf-8-sig"
                        ) as file:
                            writer = csv.writer(file)

                            if not file_exists:
                                writer.writerow(HEADER)

                            writer.writerow(row)

                        print(
                            f"已保存：{GESTURE_LABELS[selected_id]}"
                        )

                cv2.imshow("Gesture Collector", frame)

                if key == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()