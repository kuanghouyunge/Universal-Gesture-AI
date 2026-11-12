
from src.control.gesture_commands import map_gesture_to_command
from src.recognition.gesture_recognizer import GestureRecognizer
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.visualization.hand_visualizer import HandVisualizer

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

    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("无法打开摄像头，请检查摄像头连接")
        return

    print("Universal Gesture AI started.")
    print("Press Q to exit.")

    recognizer = GestureRecognizer()
    
    visualizer = HandVisualizer(
        vision.HandLandmarksConnections.HAND_CONNECTIONS
    )

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

                        # 1. 识别手势
                        result = recognizer.predict(landmarks)

                        # 2. 映射控制指令
                        command = map_gesture_to_command(
                            result["gesture"],
                            result["confidence"]
                        )

                        # 3. 获取左右手标签
                        label = results.handedness[i][0].category_name

                        if label == "Left":
                            label = "Right"
                        else:
                            label = "Left"

                        # 4. 绘制手部信息
                        frame = visualizer.draw_hand(
                            frame=frame,
                            landmarks=landmarks,
                            handedness=label,
                            gesture=result["gesture"],
                            confidence=result["confidence"],
                            command=command,
                            hand_index=i,
                        )

                        # 5. 输出调试信息
                        print("当前检测手:", label)
                        print("识别结果:", result["probabilities"])

                cv2.imshow("Universal Gesture AI - Hand Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
