
from src.control.gesture_commands import map_gesture_to_command
from src.recognition.gesture_recognizer import GestureRecognizer
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from src.visualization.hand_visualizer import HandVisualizer
from src.vision.camera_manager import CameraManager
from src.vision.hand_landmarker import HandLandmarker



def main():
    camera = CameraManager(camera_index=1)

    if not camera.open():
        return

    print("Universal Gesture AI started.")
    print("Press Q to exit.")

    recognizer = GestureRecognizer()
    
    visualizer = HandVisualizer(
        vision.HandLandmarksConnections.HAND_CONNECTIONS
    )

    try:
        with HandLandmarker() as detector:

            while True:
                success, frame = camera.read()

                if not success:
                    print("无法读取摄像头画面")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                results = detector.detect(mp_image)
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
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
