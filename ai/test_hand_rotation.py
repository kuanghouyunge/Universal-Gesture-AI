
import cv2
import mediapipe as mp
import numpy as np
import csv
import io
from contextlib import redirect_stdout

from src.vision.hand_features import (
    normalize_landmarks,
    normalize_hand_rotation,
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("摄像头是否打开:", cap.isOpened())

# 当前姿势标签
pose_label = "front"

# 已采集的数据
samples = []

pose_names = {
    ord("1"): "front",
    ord("2"): "left_30",
    ord("3"): "right_30",
}

with HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None:
            print("画面读取失败")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(
            cv2.getTickCount() / cv2.getTickFrequency() * 1000
        )

        results = detector.detect_for_video(mp_image, timestamp)

        current_samples = []

        if results.hand_landmarks:
            for i, landmarks in enumerate(results.hand_landmarks):

                raw_hand = results.handedness[i][0].category_name
                handedness = (
                    "Right" if raw_hand == "Left" else "Left"
                )

                # 屏蔽每帧的内部调试输出
                height, width = frame.shape[:2]

                with redirect_stdout(io.StringIO()):
                    raw = normalize_landmarks(landmarks)
                    local = normalize_hand_rotation(
                        landmarks,
                        handedness,
                        width,
                        height
                    )

                current_samples.append({
                    "handedness": handedness,
                    "raw": raw.copy(),
                    "local": local.copy(),
                })

        # 显示当前状态
        cv2.putText(
            frame,
            f"Pose: {pose_label}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {len(samples)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "1:Front  2:Left  3:Right  SPACE:Capture",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow("Hand Rotation Test", frame)

        key = cv2.waitKey(1) & 0xFF

        # 切换姿势标签
        if key in pose_names:
            pose_label = pose_names[key]
            print(f"\n当前姿势切换为: {pose_label}")

        # 手动采集
        elif key == ord(" "):

            if not current_samples:
                print("\n没有检测到手，无法采集")
                continue

            for item in current_samples:
                sample = {
                    "pose": pose_label,
                    "handedness": item["handedness"],
                    "raw": item["raw"],
                    "local": item["local"],
                }

                samples.append(sample)

                print("\n========== 采集成功 ==========")
                print("姿势:", pose_label)
                print("手别:", item["handedness"])
                print("原始特征:", np.round(item["raw"], 3))
                print("局部特征:", np.round(item["local"], 3))
                print("累计采集:", len(samples))

        # 保存数据
        elif key == ord("s"):

            if not samples:
                print("\n当前没有采集数据")
                continue

            with open(
                "rotation_samples.csv",
                "w",
                newline="",
                encoding="utf-8-sig"
            ) as file:

                writer = csv.writer(file)

                header = ["pose", "handedness"]
                header += [f"raw_{i}" for i in range(63)]
                header += [f"local_{i}" for i in range(63)]

                writer.writerow(header)

                for item in samples:
                    writer.writerow([
                        item["pose"],
                        item["handedness"],
                        *item["raw"],
                        *item["local"],
                    ])

            print(
                f"\n已保存 {len(samples)} 组数据："
                "rotation_samples.csv"
            )

        elif key == ord("q"):
            break

print("正在释放摄像头")
cap.release()
cv2.destroyAllWindows()
