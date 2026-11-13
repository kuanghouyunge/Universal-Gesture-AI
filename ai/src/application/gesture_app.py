
from src.control.gesture_commands import map_gesture_to_command
from src.recognition.gesture_recognizer import GestureRecognizer
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from src.visualization.hand_visualizer import HandVisualizer
from src.vision.camera_manager import CameraManager
from src.vision.hand_landmarker import HandLandmarker
from src.features.gesture_stabilizer import GestureStabilizer

class GestureApp():
    def __init__(self):
        self.camera = CameraManager(camera_index=1)

        self.recognizer = GestureRecognizer()

        self.stabilizers = {}

        self.visualizer = HandVisualizer(
            vision.HandLandmarksConnections.HAND_CONNECTIONS
        )

        self.detector = HandLandmarker()
    def process_frame(self, frame):
        # 1. 水平镜像
        frame = cv2.flip(frame, 1)

        # 2. 转换颜色空间
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # 3. 检测手部
        results = self.detector.detect(mp_image)

        # 清理当前画面中已经消失的手部缓存
        active_labels = set()

        for hand in results.handedness:
            label = hand[0].category_name

            if label == "Left":
                label = "Right"
            else:
                label = "Left"

            active_labels.add(label)

        for label in list(self.stabilizers.keys()):
            if label not in active_labels:
                del self.stabilizers[label]

        # 4. 识别并绘制
        if results.hand_landmarks:
            for i, landmarks in enumerate(results.hand_landmarks):

                result = self.recognizer.predict(landmarks)

                # 获取手部标签
                label = results.handedness[i][0].category_name

                if label == "Left":
                    label = "Right"
                else:
                    label = "Left"

                # 获取原始指令
                raw_command = map_gesture_to_command(
                    result["gesture"],
                    result["confidence"]
                )

                # 获取该只手专属的稳定器
                if label not in self.stabilizers:
                    self.stabilizers[label] = GestureStabilizer(
                        window_size=5,
                        min_count=3
                    )

                # 稳定指令
                command = self.stabilizers[label].update(raw_command)

                frame = self.visualizer.draw_hand(
                    frame=frame,
                    landmarks=landmarks,
                    handedness=label,
                    gesture=result["gesture"],
                    confidence=result["confidence"],
                    command=command,
                    hand_index=i,
                )

                print("当前检测手:", label)
                print("识别结果:", result["probabilities"])

        return frame
    def run(self):
        try:
            if not self.camera.open():
                return

            print("Universal Gesture AI started.")
            print("Press Q to exit.")

            while True:
                success, frame = self.camera.read()

                if not success:
                    print("无法读取摄像头画面")
                    break

                # 处理当前画面
                frame = self.process_frame(frame)

                # 显示画面
                cv2.imshow(
                    "Universal Gesture AI - Hand Detection",
                    frame
                )

                # 按 Q 退出
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.close()
    def close(self):
        """释放程序使用的资源。"""

        self.camera.release()

        if self.detector is not None:
            self.detector.close()
            self.detector = None

        cv2.destroyAllWindows()

        print("Universal Gesture AI stopped.")

def main():
    app = GestureApp()
    app.run()
  
if __name__ == "__main__":
    main()

