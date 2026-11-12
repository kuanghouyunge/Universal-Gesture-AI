
import cv2


class CameraManager:
    def __init__(self, camera_index=1):
        self.camera_index = camera_index
        self.cap = None

    def open(self):
        """打开摄像头。"""
        self.cap = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():
            print("无法打开摄像头，请检查摄像头连接")
            return False

        return True

    def read(self):
        """读取一帧画面。"""
        if self.cap is None:
            return False, None

        return self.cap.read()

    def release(self):
        """释放摄像头资源。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
