
from hand_features import normalize_landmarks
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"


def main():
    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开摄像头，请检查摄像头连接")
        return

    print("Universal Gesture AI started.")
    print("Press Q to exit.")

    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            timestamp_ms = 0

            while True:
                success, frame = cap.read()

                if not success:
                    print("无法读取摄像头画面")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms += 33
                results = detector.detect_for_video(mp_image, timestamp_ms)

                if results.hand_landmarks:
                    for i, landmarks in enumerate(results.hand_landmarks):
                        points = []
                        h, w, _ = frame.shape
                        features = normalize_landmarks(landmarks)

                        print(
                             f"Hand {i + 1} | "
                             f"Feature shape: {features.shape} | "
                              f"First 6 values: {features[:6]}"
                        )

                        for landmark in landmarks:
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            points.append((x, y))

                        for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                            start = points[connection.start]
                            end = points[connection.end]
                            cv2.line(frame, start, end, (255, 0, 0), 2)

                        for x, y in points:
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                        label = results.handedness[i][0].category_name

                        # 当前输入画面经过水平镜像，修正左右手标签
                        if label == "Left":
                             label = "Right"
                        else:
                             label = "Left"

                        cv2.putText(
                            frame,
                            label,
                            (points[0][0], points[0][1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Landmarks: {len(points)}",
                            (20, 40 + i * 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                cv2.imshow("Universal Gesture AI - Hand Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
